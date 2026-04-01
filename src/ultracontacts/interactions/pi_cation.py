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
from ..geometry import ring_centroids, ring_normals, centroid_to_atoms_dist_sq, centroid_to_atom_angle_deg


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
    coords: jnp.ndarray,       # (F, N, 3)
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    pc_mask: np.ndarray,       # (R, Nc) from precompute_pc_mask
) -> list[tuple]:

    dist_cut = float(geom.get("PC_CUTOFF_DIST", 6.0))
    ang_cut = float(geom.get("PC_CUTOFF_ANG", 60.0))

    R = len(groups.ring_indices)
    Nc = len(groups.cation_indices)
    if pc_mask.size == 0 or R == 0 or Nc == 0:
        return []

    F = coords.shape[0]

    # Ring geometry
    ring_xyz = coords[:, groups.ring_indices, :].reshape(F, R, 3, 3)  # (F, R, 3, 3)
    cents = np.array(ring_centroids(ring_xyz))   # (F, R, 3)
    norms = np.array(ring_normals(ring_xyz))     # (F, R, 3)

    # Cation coords
    cat_xyz = coords[:, groups.cation_indices, :]   # (F, Nc, 3)

    # Centroid→cation distances²: (F, R, Nc)
    dist_sq = np.array(centroid_to_atoms_dist_sq(
        jnp.array(cents), jnp.array(cat_xyz)
    ))

    # Angle: ring normal vs centroid→cation vector: (F, R, Nc)
    angles = np.array(centroid_to_atom_angle_deg(
        jnp.array(cents), jnp.array(norms), jnp.array(cat_xyz)
    ))

    valid = (
        (dist_sq < dist_cut ** 2) &
        (angles < ang_cut) &
        pc_mask[None, :, :]
    )

    contacts = []
    frames, ri, ci = np.nonzero(valid)
    for f, r, c in zip(frames, ri, ci):
        contacts.append((
            frame_offset + int(f), "pc",
            groups.cation_labels[c],   # cation first (matches getcontacts)
            groups.ring_labels[r],
        ))
    return contacts
