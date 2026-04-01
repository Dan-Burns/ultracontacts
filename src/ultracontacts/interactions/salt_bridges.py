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
) -> list[tuple]:
    """
    Returns list of (abs_frame, 'sb'/'sblp'/'sbpl'/'sbll', anion_label, cation_label).
    """
    cutoff = float(geom.get("SB_CUTOFF_DIST", 4.0))

    if sele_mask.size == 0:
        return []

    Na = len(groups.anion_indices)
    Nc = len(groups.cation_indices)
    F = coords.shape[0]

    # Gather anion / cation coords: (F, Na/Nc, 3)
    anion_xyz = coords[:, groups.anion_indices, :]    # (F, Na, 3)
    cation_xyz = coords[:, groups.cation_indices, :]  # (F, Nc, 3)

    # GPU Fused evaluation
    bool_mask = np.array(fused_dist_mask(anion_xyz, cation_xyz, cutoff ** 2))

    valid = bool_mask & sele_mask[None, :, :]

    contacts = []
    frames, an_pos, cat_pos = np.nonzero(valid)
    for f, ai, ci in zip(frames, an_pos, cat_pos):
        abs_frame = frame_offset + int(f)
        a_lbl = groups.anion_labels[ai]
        c_lbl = groups.cation_labels[ci]
        a_lig = bool(groups.anion_is_ligand[ai])
        c_lig = bool(groups.cation_is_ligand[ci])

        if a_lig and c_lig:
            itype = "sbll"
        elif a_lig:
            itype = "sblp"
        elif c_lig:
            itype = "sbpl"
        else:
            itype = "sb"

        contacts.append((abs_frame, itype, a_lbl, c_lbl))

    return contacts
