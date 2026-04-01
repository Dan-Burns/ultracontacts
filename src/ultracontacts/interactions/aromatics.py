"""
aromatics.py — GPU-vectorized pi-stacking and T-stacking detection.

Both interaction types (ps, ts) share the same geometric framework:
centroid distances, normal vectors, and plane-alignment angles. The only
difference is the target angle between normals.

Pi-stacking  (ps): planes nearly parallel  → normal-normal angle ≈ 0° (or 180°)
T-stacking   (ts): planes nearly perpendicular → normal-normal angle ≈ 90°

Criteria (getcontacts defaults):
  ps: centroid dist < 7.0 Å, plane angle < 30°, psi < 45°
  ts: centroid dist < 5.0 Å, |plane angle - 90| < 30°, psi < 45°
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..geometry import fused_pi_stacking_mask, fused_t_stacking_mask


def precompute_ring_pair_mask(
    groups: ChemicalGroups,
    sele1_eq_sele2: bool,
) -> np.ndarray:
    """
    Returns (R, R) bool — valid ring-ring pairs for aromatic interactions.
    Excludes self-pairs; uses upper triangle when sele1==sele2 to avoid duplicates.
    """
    R = len(groups.ring_indices)
    if R == 0:
        return np.empty((0, 0), dtype=bool)

    sele_mask = build_dual_sele_mask(
        groups.ring_in_sele1, groups.ring_in_sele2,
        groups.ring_in_sele1, groups.ring_in_sele2,
    )
    np.fill_diagonal(sele_mask, False)
    if sele1_eq_sele2:
        sele_mask &= np.triu(np.ones((R, R), dtype=bool), k=1)
    return sele_mask


def _compute_stacking(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,      # (R, R) bool
    itype: str,                      # "ps" or "ts"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Parse appropriate cutoffs
    if itype == "ps":
        dist_cut = float(geom.get("PS_CUTOFF_DIST", 7.0))
        ang_cut = float(geom.get("PS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("PS_PSI_ANG", 45.0))
    else:  # ts
        dist_cut = float(geom.get("TS_CUTOFF_DIST", 5.0))
        ang_cut = float(geom.get("TS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("TS_PSI_ANG", 45.0))

    if ring_pair_mask.size == 0 or len(groups.ring_indices) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    # Gather ring atom coords: (F, R, 3, 3)
    ring_xyz = coords[:, groups.ring_indices, :]    # (F, R*3, 3)
    F = coords.shape[0]
    ring_xyz = ring_xyz.reshape(F, R, 3, 3)         # (F, R, 3, 3)

    if itype == "ps":
        bool_mask = np.array(fused_pi_stacking_mask(ring_xyz, dist_cut ** 2, ang_cut, psi_cut))
    else:
        bool_mask = np.array(fused_t_stacking_mask(ring_xyz, dist_cut ** 2, ang_cut, psi_cut))

    frames, ri, rj = np.nonzero(valid)
    if len(frames) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    abs_frames = frames.astype(np.int32) + frame_offset
    itypes = np.full(len(frames), itype, dtype=object)
    
    lbls = np.array(groups.ring_labels, dtype=object)
    a1_arr = lbls[ri]
    a2_arr = lbls[rj]

    return abs_frames, itypes, a1_arr, a2_arr


def compute_pi_stacking(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_stacking(coords, groups, geom, frame_offset, ring_pair_mask, "ps")


def compute_t_stacking(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _compute_stacking(coords, groups, geom, frame_offset, ring_pair_mask, "ts")
