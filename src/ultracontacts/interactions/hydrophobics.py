"""
hydrophobics.py — GPU-vectorized hydrophobic contact detection.

Criterion: two hydrophobic C/S atoms within HP_CUTOFF_DIST (default 4.0 Å),
           in different residues, not in the same-chain sequential residues (res_diff filter).
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..geometry import fused_dist_mask


def precompute_hp_mask(groups: ChemicalGroups, res_diff: int) -> np.ndarray:
    """Return (Hp, Hp) bool — valid hydrophobic pairs."""
    Hp = len(groups.hp_indices)
    if Hp == 0:
        return np.empty((0, 0), dtype=bool)
    sele_mask = build_dual_sele_mask(
        groups.hp_in_sele1, groups.hp_in_sele2,
        groups.hp_in_sele1, groups.hp_in_sele2,
    )
    res_mask = build_residue_diff_mask(
        groups.hp_chain, groups.hp_resid,
        groups.hp_chain, groups.hp_resid,
        min_diff=res_diff,
    )
    # Exclude self-pairs
    np.fill_diagonal(res_mask, False)
    # Only upper triangle to avoid double-counting
    upper = np.triu(np.ones((Hp, Hp), dtype=bool), k=1)
    return sele_mask & res_mask & upper


def compute_hydrophobics(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    sele_mask: np.ndarray,    # (Hp, Hp) from precompute_hp_mask
) -> list[tuple]:
    cutoff = float(geom.get("HP_CUTOFF_DIST", 4.0))
    max_dist_sq = cutoff * cutoff

    if sele_mask.size == 0 or groups.hp_indices.size == 0:
        return []

    hp_xyz = coords[:, groups.hp_indices, :]   # (F, Hp, 3)

    # Boolean mask natively from GPU
    bool_mask = np.array(fused_dist_mask(hp_xyz, hp_xyz, max_dist_sq)) # (F, Hp, Hp)

    # Apply topology masks
    valid = bool_mask & sele_mask[None, :, :]

    contacts = []
    frames, i_pos, j_pos = np.nonzero(valid)
    for f, i, j in zip(frames, i_pos, j_pos):
        contacts.append((
            frame_offset + int(f), "hp",
            groups.hp_labels[i], groups.hp_labels[j],
        ))
    return contacts
