"""
runner.py — Main orchestration engine.

Processes trajectory frames one at a time through CuPy fused CUDA kernels.
Each kernel computes geometry + threshold + sparse extraction in one GPU pass,
never materializing the N² boolean tensor. Static data (atom indices, masks,
radii, labels) lives on GPU permanently. Only coordinates change per frame.
"""

from __future__ import annotations
import time
from typing import Optional

import numpy as np
from tqdm import tqdm
import cupy as cp
import MDAnalysis as mda

from .topology import parse_topology, ChemicalGroups
from .output import write_parquet_chunk, finalize_parquet

from .interactions.salt_bridges import precompute_sb_mask, precompute_sb_gpu, compute_salt_bridges
from .interactions.hydrophobics import precompute_hp_mask, precompute_hp_gpu, compute_hydrophobics
from .interactions.vanderwaals import precompute_vdw_mask, precompute_vdw_gpu, compute_vanderwaals
from .interactions.hbonds import precompute_hbond_mask, precompute_hb_gpu, compute_hbonds
from .interactions.aromatics import precompute_ring_pair_mask, precompute_ring_gpu, compute_pi_stacking, compute_t_stacking
from .interactions.pi_cation import precompute_pc_mask, precompute_pc_gpu, compute_pi_cation

# ---------------------------------------------------------------------------
# Default geometric criteria — mirror getcontacts defaults
# ---------------------------------------------------------------------------
DEFAULT_GEOM = {
    "SB_CUTOFF_DIST": 4.0,
    "PC_CUTOFF_DIST": 6.0,
    "PC_CUTOFF_ANG": 60.0,
    "PS_CUTOFF_DIST": 7.0,
    "PS_CUTOFF_ANG": 30.0,
    "PS_PSI_ANG": 45.0,
    "TS_CUTOFF_DIST": 5.0,
    "TS_CUTOFF_ANG": 30.0,
    "TS_PSI_ANG": 45.0,
    "HBOND_CUTOFF_DIST": 3.5,
    "HBOND_CUTOFF_ANG": 110.0,   # VMD uses 70° deviation from linear = D-H-A > 110°
    "HBOND_RES_DIFF": 1,
    "VDW_EPSILON": 0.5,
    "VDW_RES_DIFF": 2,
    "HP_CUTOFF_DIST": 4.0,
    "HP_RES_DIFF": 2,
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
    openmm_system: Optional[str] = None,   # deprecated — use bond_topology
    bond_topology: Optional[str] = None,
    n_gpus: Optional[int] = None,
    chunk_size: int = 25,  # kept for CLI compat, unused in CuPy path
) -> None:
    t0 = time.perf_counter()

    if sele2 is None:
        sele2 = sele1
    sele1_eq_sele2 = (sele1 == sele2)

    if "all" in itypes:
        itypes = list(ALL_ITYPES)
    itypes = [it.lower() for it in itypes]

    geom = dict(DEFAULT_GEOM)
    if geom_criteria:
        geom.update(geom_criteria)

    # Select CUDA device
    device = cp.cuda.Device(0)
    print(f"[ultracontacts] Using CUDA device: {device.id} ({cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()})")

    # ---- Load topology ----
    print(f"[ultracontacts] Loading topology: {topology}")
    if trajectory:
        u = mda.Universe(topology, trajectory)
    else:
        u = mda.Universe(topology)

    # ---- Load bond topology from external file ----
    bond_topo_path = bond_topology or openmm_system
    if bond_topo_path:
        from .topology import parse_bond_topology
        print(f"[ultracontacts] Loading bonds from: {bond_topo_path}")
        try:
            bonds = parse_bond_topology(bond_topo_path, max_atoms=len(u.atoms))
            if len(bonds) > 0:
                u.add_TopologyAttr('bonds', bonds)
                print(f"[ultracontacts] Assigned {len(bonds)} bonds")
            else:
                print("[ultracontacts] Warning: topology parsed but 0 bonds found")
        except Exception as e:
            print(f"[ultracontacts] Error parsing bond topology: {e}")

    if "hb" in itypes:
        n_bonds = 0
        try:
            if hasattr(u, "bonds"):
                n_bonds = len(u.bonds)
        except Exception:
            pass
        # A reasonable protein has ~1 bond per atom; 5 CONECT records for
        # 20K atoms is clearly incomplete
        if n_bonds < len(u.atoms) // 2:
            if n_bonds > 0:
                print(f"[ultracontacts] Warning: only {n_bonds} bonds for {len(u.atoms)} atoms — incomplete bond info")
            print("[ultracontacts] Guessing bonds (this may take a moment)…")
            try:
                u.atoms.guess_bonds()
                n_bonds_new = len(u.bonds)
                print(f"[ultracontacts] Guessed {n_bonds_new} bonds")
                if n_bonds_new < len(u.atoms) // 2:
                    print("[ultracontacts] ⚠ Bond guessing produced few bonds — "
                          "H-bond detection will be unreliable.\n"
                          "  → Pass --bond-topology <system.xml|.prmtop|.psf|…> for accurate results.")
            except Exception as e:
                print(f"[ultracontacts] Warning: bond guessing failed: {e}\n"
                      "  → Pass --bond-topology <system.xml|.prmtop|.psf|…> for H-bond detection.")

    # ---- Parse chemical groups ----
    print("[ultracontacts] Parsing chemical groups…")
    groups = parse_topology(u, sele1, sele2)
    _report_groups(groups, itypes)

    # ---- Pre-compute static masks (CPU) and upload to GPU (once) ----
    print("[ultracontacts] Uploading static data to GPU…")
    gpu = {}
    if "sb" in itypes:
        mask = precompute_sb_mask(groups, sele1_eq_sele2)
        gpu["sb"] = precompute_sb_gpu(groups, mask)
    if "hp" in itypes:
        mask = precompute_hp_mask(groups, int(geom.get("HP_RES_DIFF", 2)))
        gpu["hp"] = precompute_hp_gpu(groups, mask)
    if "vdw" in itypes:
        pair_mask, _ = precompute_vdw_mask(groups, int(geom.get("VDW_RES_DIFF", 2)), sele1_eq_sele2)
        gpu["vdw"] = precompute_vdw_gpu(groups, pair_mask)
    if "hb" in itypes:
        mask = precompute_hbond_mask(groups, int(geom.get("HBOND_RES_DIFF", 1)))
        gpu["hb"] = precompute_hb_gpu(groups, mask)
    if "ps" in itypes or "ts" in itypes:
        mask = precompute_ring_pair_mask(groups, sele1_eq_sele2)
        gpu["rings"] = precompute_ring_gpu(groups, mask)
    if "pc" in itypes:
        mask = precompute_pc_mask(groups)
        gpu["pc"] = precompute_pc_gpu(groups, mask)

    # ---- Frame range ----
    n_frames_total = len(u.trajectory) if trajectory else 1
    beg = max(0, beg)
    end = min(n_frames_total - 1, end) if end is not None else n_frames_total - 1
    frames_to_process = list(range(beg, end + 1, stride))
    n_frames = len(frames_to_process)
    print(f"[ultracontacts] Processing {n_frames} frames (beg={beg}, end={end}, stride={stride})")

    # ---- Per-frame processing with buffered writes ----
    WRITE_EVERY = 100
    parquet_writer = None
    buf_f, buf_it, buf_a1, buf_a2 = [], [], [], []
    n_contacts_total = 0

    def _add(res):
        if len(res[0]) > 0:
            buf_f.append(res[0])
            buf_it.append(res[1])
            buf_a1.append(res[2])
            buf_a2.append(res[3])

    def _flush():
        nonlocal parquet_writer
        if buf_f:
            contacts = (np.concatenate(buf_f), np.concatenate(buf_it),
                        np.concatenate(buf_a1), np.concatenate(buf_a2))
            parquet_writer = write_parquet_chunk(
                contacts, output, parquet_writer, itypes, beg, end, stride
            )
            buf_f.clear(); buf_it.clear(); buf_a1.clear(); buf_a2.clear()

    with tqdm(total=n_frames, unit="frame", desc="Computing contacts") as pbar:
        for fi, abs_frame in enumerate(frames_to_process):
            # Upload coordinates for this frame (flat float32 on GPU)
            u.trajectory[abs_frame]
            coords_gpu = cp.asarray(u.atoms.positions.ravel(), dtype=cp.float32)

            # Fused kernel calls — each returns only sparse hit indices
            if "sb" in itypes:  _add(compute_salt_bridges(coords_gpu, gpu["sb"], geom, abs_frame))
            if "hp" in itypes:  _add(compute_hydrophobics(coords_gpu, gpu["hp"], geom, abs_frame))
            if "vdw" in itypes: _add(compute_vanderwaals(coords_gpu, gpu["vdw"], geom, abs_frame))
            if "hb" in itypes:  _add(compute_hbonds(coords_gpu, gpu["hb"], geom, abs_frame))
            if "ps" in itypes:  _add(compute_pi_stacking(coords_gpu, gpu["rings"], geom, abs_frame))
            if "ts" in itypes:  _add(compute_t_stacking(coords_gpu, gpu["rings"], geom, abs_frame))
            if "pc" in itypes:  _add(compute_pi_cation(coords_gpu, gpu["pc"], geom, abs_frame))

            pbar.update(1)

            # Periodic flush + update postfix stats
            if (fi + 1) % WRITE_EVERY == 0 or fi == n_frames - 1:
                n_buf = sum(len(a) for a in buf_f) if buf_f else 0
                n_contacts_total += n_buf
                _flush()
                pbar.set_postfix(contacts=f"{n_contacts_total:,}")

    finalize_parquet(parquet_writer, output)

    elapsed = time.perf_counter() - t0
    print(f"[ultracontacts] Done — {n_contacts_total:,} contacts — output: {output}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_groups(groups: ChemicalGroups, itypes: list[str]):
    print(f"  Aromatic rings  : {len(groups.ring_indices)}")
    print(f"  H-bond donors   : {len(groups.donor_indices)}")
    print(f"  H-bond acceptors: {len(groups.acceptor_indices)}")
    print(f"  Anions          : {len(groups.anion_indices)}")
    print(f"  Cations         : {len(groups.cation_indices)}")
    print(f"  Hydrophobic     : {len(groups.hp_indices)}")
    print(f"  VdW atoms (s1)  : {len(groups.vdw_indices1)}")
    print(f"  VdW atoms (s2)  : {len(groups.vdw_indices2)}")
