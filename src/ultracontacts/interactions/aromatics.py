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
from ..geometry import ring_centroids, ring_normals, angle_between_normals_deg, psi_angle_deg


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


def _compute_aromatics(
    coords: jnp.ndarray,          # (F, N, 3)
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,   # (R, R)
    itype: str,                   # "ps" or "ts"
) -> list[tuple]:

    R = len(groups.ring_indices)
    if R == 0 or ring_pair_mask.size == 0:
        return []

    if itype == "ps":
        dist_cut = float(geom.get("PS_CUTOFF_DIST", 7.0))
        ang_cut = float(geom.get("PS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("PS_PSI_ANG", 45.0))
    else:
        dist_cut = float(geom.get("TS_CUTOFF_DIST", 5.0))
        ang_cut = float(geom.get("TS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("TS_PSI_ANG", 45.0))

    # Gather ring atom coords: (F, R, 3, 3)
    ring_xyz = coords[:, groups.ring_indices, :]    # (F, R*3, 3)
    F = coords.shape[0]
    ring_xyz = ring_xyz.reshape(F, R, 3, 3)         # (F, R, 3, 3)

    # Centroids and normals: (F, R, 3)
    cents = np.array(ring_centroids(ring_xyz))       # (F, R, 3)
    norms = np.array(ring_normals(ring_xyz))         # (F, R, 3)

    # Pairwise centroid distances: (F, R, R)
    diff = cents[:, :, None, :] - cents[:, None, :, :]  # (F, R, R, 3)
    cent_dists = np.sqrt(np.sum(diff * diff, axis=-1))   # (F, R, R)

    # Plane-alignment angle: (F, R, R)
    # angle_between_normals operates on (..., 3) — we compute for all pairs
    n1 = norms[:, :, None, :]   # (F, R, 1, 3)
    n2 = norms[:, None, :, :]   # (F, 1, R, 3)
    cos_angle = np.clip(np.abs(np.sum(n1 * n2, axis=-1)), 0.0, 1.0)  # (F, R, R)
    plane_angle = np.degrees(np.arccos(cos_angle))   # (F, R, R)

    # Psi angle: min of psi(ring1→ring2) and psi(ring2→ring1)
    vec = diff                                       # (F, R, R, 3)  c2 - c1
    vec_norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec_norm = np.where(vec_norm < 1e-8, 1.0, vec_norm)
    vec_unit = vec / vec_norm                        # (F, R, R, 3)
    psi1 = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(vec_unit * n1, axis=-1)), 0.0, 1.0)))  # (F, R, R)
    psi2 = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(vec_unit * n2, axis=-1)), 0.0, 1.0)))  # (F, R, R)
    psi = np.minimum(psi1, psi2)

    # Build final contact mask
    dist_ok = cent_dists < dist_cut                          # (F, R, R)

    if itype == "ps":
        ang_ok = plane_angle < ang_cut
    else:  # ts
        ang_ok = np.abs(plane_angle - 90.0) < ang_cut

    psi_ok = psi < psi_cut

    valid = dist_ok & ang_ok & psi_ok & ring_pair_mask[None, :, :]

    contacts = []
    frames, ri, rj = np.nonzero(valid)
    for f, i, j in zip(frames, ri, rj):
        contacts.append((
            frame_offset + int(f), itype,
            groups.ring_labels[i], groups.ring_labels[j],
        ))
    return contacts


def compute_pi_stacking(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,
) -> list[tuple]:
    return _compute_aromatics(coords, groups, geom, frame_offset, ring_pair_mask, "ps")


def compute_t_stacking(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    ring_pair_mask: np.ndarray,
) -> list[tuple]:
    return _compute_aromatics(coords, groups, geom, frame_offset, ring_pair_mask, "ts")
