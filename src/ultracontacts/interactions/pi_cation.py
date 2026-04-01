"""
pi_cation.py — GPU-vectorized pi-cation interaction detection.

Criterion (getcontacts defaults):
  - cation to ring centroid distance < PC_CUTOFF_DIST (default 6.0 Å)
  - angle between ring normal and centroid→cation vector < PC_CUTOFF_ANG (default 60°)
    (i.e., cation is roughly on the face of the ring)
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..geometry import fused_pi_cation_mask


def precompute_pc_mask(groups: ChemicalGroups) -> np.ndarray:
    """
    Returns (R, Nc) bool — valid ring-cation pairs.
    """
    R = len(groups.ring_indices)
    Nc = len(groups.cation_indices)
    if R == 0 or Nc == 0:
        return np.empty((0, 0), dtype=bool)

    # A pi-cation contact is valid if ring and cation cross sele1/sele2
    # ring_in_sele acts as a proxy for "ring" membership
    return build_dual_sele_mask(
        groups.ring_in_sele1, groups.ring_in_sele2,
        groups.cation_in_sele1, groups.cation_in_sele2,
    )


def compute_pi_cation(
    coords: jnp.ndarray,         # (F, N, 3) on-device
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    sele_mask: np.ndarray,        # (R, C) bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    dist_cutoff = float(geom.get("PC_CUTOFF_DIST", 6.0))
    ang_cutoff = float(geom.get("PC_CUTOFF_ANG", 60.0))

    if sele_mask.size == 0 or len(groups.ring_indices) == 0 or len(groups.cation_indices) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    F = coords.shape[0]
    R = len(groups.ring_indices)
    Nc = len(groups.cation_indices)

    # Ring geometry
    ring_xyz = coords[:, groups.ring_indices, :].reshape(F, R, 3, 3)  # (F, R, 3, 3)

    # Cation coords
    cat_xyz = coords[:, groups.cation_indices, :]   # (F, Nc, 3)

    # Boolean mask evaluation in a single fused JAX GPU kernel
    bool_mask = np.array(fused_pi_cation_mask(
        ring_xyz, cat_xyz,
        dist_cutoff ** 2, 1000.0, ang_cutoff
    ))  # (F, R, C)

    valid = bool_mask & sele_mask[None, :, :]

    frames, rin_pos, cat_pos = np.nonzero(valid)
    if len(frames) == 0:
        return (np.empty(0, dtype=np.int32), np.empty(0, dtype=object), np.empty(0, dtype=object), np.empty(0, dtype=object))

    abs_frames = frames.astype(np.int32) + frame_offset
    itypes = np.full(len(frames), "pc", dtype=object)
    
    r_lbls = np.array(groups.ring_labels, dtype=object)[rin_pos]
    c_lbls = np.array(groups.cation_labels, dtype=object)[cat_pos]

    return abs_frames, itypes, c_lbls, r_lbls
