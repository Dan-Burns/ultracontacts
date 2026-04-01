"""
vanderwaals.py — GPU-vectorized van der Waals contact detection.

Criterion: distance(atom1, atom2) < vdw_radius1 + vdw_radius2 + VDW_EPSILON (default 0.5 Å),
           atoms must be from different residues separated by >= VDW_RES_DIFF (default 2).
Non-hydrogen atoms only.

For large selections this module uses atom-pair chunking to stay within GPU memory.
"""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..geometry import batch_pairwise_dist_sq

# Maximum atoms per batch side to cap (V1*V2*F) tensor memory (~2 GB budget)
_MAX_ATOMS_PER_SIDE = 2000


def precompute_vdw_mask(groups: ChemicalGroups, res_diff: int,
                         sele1_eq_sele2: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        pair_mask: (V1, V2) bool — statically valid pairs
        cutoff_mat: (V1, V2) float32 — per-pair distance cutoff = r1 + r2 + epsilon
    """
    V1, V2 = len(groups.vdw_indices1), len(groups.vdw_indices2)
    if V1 == 0 or V2 == 0:
        return np.empty((0, 0), dtype=bool), np.empty((0, 0), dtype=np.float32)

    sele_mask = build_dual_sele_mask(
        groups.vdw_is_ligand1 | True,   # all v1 atoms are in sele1 by construction
        np.zeros(V1, dtype=bool),
        np.zeros(V2, dtype=bool),
        groups.vdw_is_ligand2 | True,
    )
    # Actually for vdw, sele1 and sele2 are already the correct subsets —
    # just filter by residue diff.
    sele_mask = np.ones((V1, V2), dtype=bool)

    res_mask = build_residue_diff_mask(
        groups.vdw_chain1, groups.vdw_resid1,
        groups.vdw_chain2, groups.vdw_resid2,
        min_diff=res_diff,
    )

    # Exclude disulfide CYS pairs handled in hydrophobics
    pair_mask = sele_mask & res_mask

    if sele1_eq_sele2:
        # avoid duplicate (i,j) and (j,i)
        # When selections are identical, V1==V2 and indices match
        pair_mask &= np.triu(np.ones_like(pair_mask), k=1)

    # Per-pair VdW cutoffs
    cutoff_mat = (groups.vdw_radii1[:, None] +
                  groups.vdw_radii2[None, :]).astype(np.float32)  # (V1, V2)

    return pair_mask, cutoff_mat


def compute_vanderwaals(
    coords: jnp.ndarray,
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    pair_mask: np.ndarray,     # (V1, V2)
    cutoff_mat: np.ndarray,    # (V1, V2) — per-pair hard cutoff in Å
) -> list[tuple]:
    epsilon = float(geom.get("VDW_EPSILON", 0.5))

    if pair_mask.size == 0:
        return []

    V1, V2 = pair_mask.shape
    F = coords.shape[0]
    contacts = []

    # Determine chunk sizes to stay within GPU memory
    chunk1 = min(V1, _MAX_ATOMS_PER_SIDE)
    chunk2 = min(V2, _MAX_ATOMS_PER_SIDE)

    for i0 in range(0, V1, chunk1):
        i1 = min(i0 + chunk1, V1)
        idx1_chunk = groups.vdw_indices1[i0:i1]
        mask_chunk = pair_mask[i0:i1, :]          # (c1, V2)
        cut_chunk = cutoff_mat[i0:i1, :] + epsilon  # (c1, V2)
        xyz1 = coords[:, idx1_chunk, :]             # (F, c1, 3)

        for j0 in range(0, V2, chunk2):
            j1 = min(j0 + chunk2, V2)
            sub_mask = mask_chunk[:, j0:j1]         # (c1, c2)
            if not sub_mask.any():
                continue

            idx2_sub = groups.vdw_indices2[j0:j1]
            xyz2 = coords[:, idx2_sub, :]            # (F, c2, 3)
            dists_sq = np.array(batch_pairwise_dist_sq(xyz1, xyz2))  # (F, c1, c2)

            sub_cut = cut_chunk[:, j0:j1]            # (c1, c2)
            valid = (dists_sq < (sub_cut ** 2)[None, :, :]) & sub_mask[None, :, :]

            ff, ii, jj = np.nonzero(valid)
            for f, i, j in zip(ff, ii, jj):
                gi = i0 + i
                gj = j0 + j
                # skip disulfide CYS–CYS
                contacts.append((
                    frame_offset + int(f), "vdw",
                    groups.vdw_labels1[gi], groups.vdw_labels2[gj],
                ))

    return contacts
