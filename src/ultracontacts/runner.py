"""
runner.py — Main orchestration engine.

Streams trajectory frames via MDAnalysis in chunks, dispatches GPU computation
per chunk (optionally cycling across multiple CUDA devices), and accumulates contacts.

Multi-GPU: each chunk is placed on a different device via round-robin.
           Uses threading to overlap host↔device transfers with computation.
"""

from __future__ import annotations
import time
import os
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import MDAnalysis as mda

_worker_u = None

def _init_worker(topo_path, traj_path):
    global _worker_u
    _worker_u = mda.Universe(topo_path, traj_path)

def _read_traj_chunk_wrapper(frames):
    global _worker_u
    out = []
    for f in frames:
        _worker_u.trajectory[f]
        out.append(_worker_u.atoms.positions.copy())
    return np.stack(out, axis=0), frames
from MDAnalysis.coordinates.memory import MemoryReader

from .topology import parse_topology, ChemicalGroups
from .output import write_parquet_chunk, finalize_parquet

from .interactions.salt_bridges import precompute_sb_mask, compute_salt_bridges
from .interactions.hydrophobics import precompute_hp_mask, compute_hydrophobics
from .interactions.vanderwaals import precompute_vdw_mask, compute_vanderwaals
from .interactions.hbonds import precompute_hbond_mask, compute_hbonds
from .interactions.aromatics import precompute_ring_pair_mask, compute_pi_stacking, compute_t_stacking
from .interactions.pi_cation import precompute_pc_mask, compute_pi_cation

# ---------------------------------------------------------------------------
# Default geometric criteria — mirror getcontacts defaults
# ---------------------------------------------------------------------------
DEFAULT_GEOM = {
    "SB_CUTOFF_DIST": 4.0,       # Å   anion-cation
    "PC_CUTOFF_DIST": 6.0,       # Å   cation-ring centroid
    "PC_CUTOFF_ANG": 60.0,       # °   normal-vs-centroid→cation
    "PS_CUTOFF_DIST": 7.0,       # Å   ring centroid-centroid
    "PS_CUTOFF_ANG": 30.0,       # °   plane-alignment (from 0°)
    "PS_PSI_ANG": 45.0,          # °   psi
    "TS_CUTOFF_DIST": 5.0,       # Å
    "TS_CUTOFF_ANG": 30.0,       # °   plane-alignment (from 90°)
    "TS_PSI_ANG": 45.0,          # °
    "HBOND_CUTOFF_DIST": 3.5,    # Å   D···A
    "HBOND_CUTOFF_ANG": 150.0,   # °   min D-H···A (getcontacts uses 180-70=110°;
                                  #     we default stricter: 150°)
    "HBOND_RES_DIFF": 1,         # min residue separation
    "VDW_EPSILON": 0.5,           # Å   added to sum of VdW radii
    "VDW_RES_DIFF": 2,            # min residue separation
    "HP_CUTOFF_DIST": 4.0,        # Å
    "HP_RES_DIFF": 2,             # min residue separation
}

ALL_ITYPES = {"sb", "pc", "ps", "ts", "hb", "vdw", "hp"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_contacts(
    topology: str,
    trajectory: Optional[str],
    itypes: list[str] = ("hb", "sb", "ps", "ts", "pc", "vdw", "hp"),
    geom_criteria: Optional[dict] = None,
    sele1: str = "protein",
    sele2: Optional[str] = None,
    beg: int = 0,
    end: Optional[int] = None,
    stride: int = 1,
    output: str = "contacts.parquet",
    openmm_system: Optional[str] = None,
    n_gpus: Optional[int] = None,
    chunk_size: int = 500,
) -> None:
    """
    Compute molecular contacts and write them to a Parquet file.

    Parameters
    ----------
    topology : path to topology (.pdb, .psf, .prmtop, …)
    trajectory : path to trajectory (.dcd, .xtc, .nc, …), or None for single frame
    itypes : list of interaction type codes to compute
    geom_criteria : override any DEFAULT_GEOM value
    sele1 : MDAnalysis selection string for group 1
    sele2 : MDAnalysis selection string for group 2 (defaults to sele1)
    beg, end, stride : frame range
    output : path to output .parquet file
    openmm_system : path to OpenMM system.xml to import bonds from (optional)
    n_gpus : number of CUDA devices to use (default: all available)
    chunk_size : frames per GPU dispatch
    """
    t0 = time.perf_counter()

    if sele2 is None:
        sele2 = sele1
    sele1_eq_sele2 = (sele1 == sele2)

    # Resolve itypes
    if "all" in itypes:
        itypes = list(ALL_ITYPES)
    itypes = [it.lower() for it in itypes]

    # Merge geometry criteria
    geom = dict(DEFAULT_GEOM)
    if geom_criteria:
        geom.update(geom_criteria)

    # Select CUDA devices
    try:
        all_devices = jax.devices("cuda")
    except Exception:
        all_devices = jax.devices()
    if n_gpus is not None:
        all_devices = all_devices[:n_gpus]
    devices = all_devices
    print(f"[ultracontacts] Using {len(devices)} device(s): {devices}")

    # ---- Load topology ----
    print(f"[ultracontacts] Loading topology: {topology}")
    if trajectory:
        u = mda.Universe(topology, trajectory)
    else:
        u = mda.Universe(topology)

    if openmm_system:
        from .topology import parse_openmm_bonds
        print(f"[ultracontacts] Loading bonds from OpenMM system: {openmm_system}")
        try:
            bonds = parse_openmm_bonds(openmm_system, max_atoms=len(u.atoms))
            if len(bonds) > 0:
                u.add_TopologyAttr('bonds', bonds)
                print(f"[ultracontacts] Assigned {len(bonds)} bonds from XML")
            else:
                print("[ultracontacts] Warning: XML parsed but 0 bonds found")
        except Exception as e:
            print(f"[ultracontacts] Error parsing OpenMM system: {e}")

    # Guess bonds if not present (needed for H-bond donors)
    if "hb" in itypes:
        has_bonds = False
        try:
            if hasattr(u, "bonds") and len(u.bonds) > 0:
                has_bonds = True
        except Exception:
            pass
            
        if not has_bonds:
            print("[ultracontacts] Guessing bonds (no bond info in topology)…")
            try:
                u.atoms.guess_bonds()
            except Exception as e:
                print(f"[ultracontacts] Warning: bond guessing failed: {e}")

    # ---- Parse chemical groups ----
    print("[ultracontacts] Parsing chemical groups…")
    groups = parse_topology(u, sele1, sele2)
    _report_groups(groups, itypes)

    # ---- Pre-compute static masks ----
    masks = _precompute_masks(groups, geom, sele1_eq_sele2, itypes)

    # ---- Auto-tune Chunk Size ----
    max_pairs = 1
    if "sb" in itypes: max_pairs = max(max_pairs, len(groups.anion_indices) * len(groups.cation_indices))
    if "hp" in itypes: max_pairs = max(max_pairs, len(groups.hp_indices) ** 2)
    if "hb" in itypes: max_pairs = max(max_pairs, len(groups.donor_indices) * len(groups.acceptor_indices))
    if "ps" in itypes or "ts" in itypes: max_pairs = max(max_pairs, len(groups.ring_indices) ** 2)
    if "pc" in itypes: max_pairs = max(max_pairs, len(groups.ring_indices) * len(groups.cation_indices))
    # VdW is internally atom-chunked (max 2000x2000 = 4M elements per block), so we cap its contribution
    if "vdw" in itypes: max_pairs = max(max_pairs, min(len(groups.vdw_indices1) * len(groups.vdw_indices2), 4_000_000))

    # XLA limit is ~2GB array size per tensor. Because we now use Fused Boolean Masks,
    # the returned array is 1-byte per element instead of 4-byte float32 elements.
    # We can comfortably limit chunking combinations to 500 Million pairs (~500 MB output).
    safe_chunk_size = max(1, 500_000_000 // max_pairs)
    if chunk_size > safe_chunk_size:
        print(f"[ultracontacts] Dense topology detected (Max pairwise density: {max_pairs:,} pairs/frame).")
        print(f"[ultracontacts] Auto-tuning chunk_size from {chunk_size} -> {safe_chunk_size} GPU batch to prevent bitmask OOM limits.")
        chunk_size = safe_chunk_size

    # ---- Frame range ----
    n_frames_total = len(u.trajectory) if trajectory else 1
    beg = max(0, beg)
    end = min(n_frames_total - 1, end) if end is not None else n_frames_total - 1
    frames_to_process = list(range(beg, end + 1, stride))
    n_frames = len(frames_to_process)
    print(f"[ultracontacts] Processing {n_frames} frames (beg={beg}, end={end}, stride={stride})")

    # ---- Stream and dispatch ----
    parquet_writer = [None]  # mutable container for the ParquetWriter
    write_lock = threading.Lock()

    def _process_chunk(chunk_xyz: np.ndarray, frame_offsets: list[int], device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run all enabled interaction types on a coordinate chunk."""
        jax_coords = jax.device_put(jnp.array(chunk_xyz), device)
        frame_offset = frame_offsets[0]
        
        f_arrs, it_arrs, a1_arrs, a2_arrs = [], [], [], []
        def _add(res):
            if len(res[0]) > 0:
                f_arrs.append(res[0])
                it_arrs.append(res[1])
                a1_arrs.append(res[2])
                a2_arrs.append(res[3])

        if "sb" in itypes: _add(compute_salt_bridges(jax_coords, groups, geom, frame_offset, masks["sb"]))
        if "hp" in itypes: _add(compute_hydrophobics(jax_coords, groups, geom, frame_offset, masks["hp"]))
        if "vdw" in itypes: _add(compute_vanderwaals(jax_coords, groups, geom, frame_offset, masks["vdw_pair"], masks["vdw_cut"]))
        if "hb" in itypes: _add(compute_hbonds(jax_coords, groups, geom, frame_offset, masks["hb"]))
        if "ps" in itypes: _add(compute_pi_stacking(jax_coords, groups, geom, frame_offset, masks["rings"]))
        if "ts" in itypes: _add(compute_t_stacking(jax_coords, groups, geom, frame_offset, masks["rings"]))
        if "pc" in itypes: _add(compute_pi_cation(jax_coords, groups, geom, frame_offset, masks["pc"]))
        
        if not f_arrs:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object)
            
        return (
            np.concatenate(f_arrs),
            np.concatenate(it_arrs),
            np.concatenate(a1_arrs),
            np.concatenate(a2_arrs),
        )

    futures = []
    chunked_frames = [frames_to_process[i:i + chunk_size] for i in range(0, len(frames_to_process), chunk_size)]

    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as gpu_executor:
        if trajectory:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            max_workers = min(os.cpu_count() or 4, 8)
            print(f"[ultracontacts] Launching Asynchronous Dataloader ({max_workers} processes)...")
            
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_init_worker, initargs=(str(topology), str(trajectory))) as loader_executor:
                device_idx = 0
                for chunk_xyz, chunk_frames in loader_executor.map(_read_traj_chunk_wrapper, chunked_frames):
                    device = devices[device_idx % len(devices)]
                    device_idx += 1
                    
                    # Submit to GPU thread and append to queue
                    fut = gpu_executor.submit(_process_chunk, chunk_xyz, chunk_frames, device)
                    futures.append((fut, chunk_frames[0]))
                    
                    # Interleaved PyArrow Writer (Check the oldest future in the pipeline to write in exact chronological order without hanging memory)
                    while len(futures) > 0 and futures[0][0].done():
                        done_fut, _ = futures.pop(0)
                        contacts = done_fut.result()
                        if len(contacts[0]) > 0:
                            with write_lock:
                                parquet_writer[0] = write_parquet_chunk(
                                    contacts, output, parquet_writer[0], itypes, beg, end, stride
                                )

            # Flush the remaining futures sequentially
            for fut, _ in futures:
                contacts = fut.result()
                if len(contacts[0]) > 0:
                    with write_lock:
                        parquet_writer[0] = write_parquet_chunk(
                            contacts, output, parquet_writer[0], itypes, beg, end, stride
                        )
        else:
            # Single PDB Evaluation
            chunk_xyz = np.stack([u.atoms.positions.copy()])
            chunk_frames = [0]
            device = devices[0]
            fut = gpu_executor.submit(_process_chunk, chunk_xyz, chunk_frames, device)
            contacts = fut.result()
            if len(contacts[0]) > 0:
                 parquet_writer[0] = write_parquet_chunk(
                     contacts, output, parquet_writer[0], itypes, beg, end, stride
                 )

    finalize_parquet(parquet_writer[0], output)

    elapsed = time.perf_counter() - t0
    print(f"[ultracontacts] Done — output: {output}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _precompute_masks(groups, geom, sele1_eq_sele2, itypes):
    masks = {}
    if "sb" in itypes:
        masks["sb"] = precompute_sb_mask(groups, sele1_eq_sele2)
    if "hp" in itypes:
        masks["hp"] = precompute_hp_mask(groups, int(geom.get("HP_RES_DIFF", 2)))
    if "vdw" in itypes:
        pair, cut = precompute_vdw_mask(groups, int(geom.get("VDW_RES_DIFF", 2)), sele1_eq_sele2)
        masks["vdw_pair"] = pair
        masks["vdw_cut"] = cut
    if "hb" in itypes:
        masks["hb"] = precompute_hbond_mask(groups, int(geom.get("HBOND_RES_DIFF", 1)))
    if "ps" in itypes or "ts" in itypes:
        masks["rings"] = precompute_ring_pair_mask(groups, sele1_eq_sele2)
    if "pc" in itypes:
        masks["pc"] = precompute_pc_mask(groups)
    return masks


def _report_groups(groups: ChemicalGroups, itypes: list[str]):
    print(f"  Aromatic rings  : {len(groups.ring_indices)}")
    print(f"  H-bond donors   : {len(groups.donor_indices)}")
    print(f"  H-bond acceptors: {len(groups.acceptor_indices)}")
    print(f"  Anions          : {len(groups.anion_indices)}")
    print(f"  Cations         : {len(groups.cation_indices)}")
    print(f"  Hydrophobic     : {len(groups.hp_indices)}")
    print(f"  VdW atoms (s1)  : {len(groups.vdw_indices1)}")
    print(f"  VdW atoms (s2)  : {len(groups.vdw_indices2)}")
