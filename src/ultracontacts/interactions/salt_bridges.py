"""
salt_bridges.py — GPU-vectorized salt bridge detection.

Criterion: distance between anion atom and cation atom < SB_CUTOFF_DIST (default 4.0 Å).
No angle criterion (matches getcontacts).
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..geometry import fused_dist_mask

# Cross-sele validity is pre-computed once outside the frame loop.


def precompute_sb_mask(groups: ChemicalGroups, sele1_eq_sele2: bool) -> np.ndarray:
    """Return (Na, Nc) bool array — valid anion-cation pairs for this sele combo."""
    if len(groups.anion_indices) == 0 or len(groups.cation_indices) == 0:
        return np.empty((0, 0), dtype=bool)
    return build_dual_sele_mask(
        groups.anion_in_sele1, groups.anion_in_sele2,
        groups.cation_in_sele1, groups.cation_in_sele2,
    )


def compute_salt_bridges(
    coords: jnp.ndarray,          # (F, N, 3)  on-device
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    sele_mask: np.ndarray,         # (Na, Nc) bool — from precompute_sb_mask
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (frames_arr, itypes_arr, atom1_arr, atom2_arr).
    """
    cutoff = float(geom.get("SB_CUTOFF_DIST", 4.0))

    if sele_mask.size == 0 or len(groups.anion_indices) == 0 or len(groups.cation_indices) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    Na = len(groups.anion_indices)
    Nc = len(groups.cation_indices)
    F = coords.shape[0]

    # Gather anion / cation coords: (F, Na/Nc, 3)
    anion_xyz = coords[:, groups.anion_indices, :]    # (F, Na, 3)
    cation_xyz = coords[:, groups.cation_indices, :]  # (F, Nc, 3)

    # GPU Fused evaluation
    bool_mask = np.array(fused_dist_mask(anion_xyz, cation_xyz, cutoff ** 2))

    valid = bool_mask & sele_mask[None, :, :]

    frames, an_pos, cat_pos = np.nonzero(valid)
    if len(frames) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    abs_frames = frames.astype(np.int32) + frame_offset
    
    a_lbls = np.array(groups.anion_labels, dtype=object)[an_pos]
    c_lbls = np.array(groups.cation_labels, dtype=object)[cat_pos]
    
    a_lig = np.array(groups.anion_is_ligand, dtype=bool)[an_pos]
    c_lig = np.array(groups.cation_is_ligand, dtype=bool)[cat_pos]

    itypes = np.full(len(frames), "sb", dtype=object)
    itypes[a_lig & ~c_lig] = "sblp"
    itypes[~a_lig & c_lig] = "sbpl"
    itypes[a_lig & c_lig]  = "sbll"

    return abs_frames, itypes, a_lbls, c_lbls
