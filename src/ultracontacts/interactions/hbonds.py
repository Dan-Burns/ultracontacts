"""
hbonds.py — GPU-vectorized hydrogen bond detection.

Requires explicit H in topology.

Criteria:
  - D···A distance < HBOND_CUTOFF_DIST  (default 3.5 Å)
  - D-H···A angle  > HBOND_CUTOFF_ANG   (default 150°; getcontacts uses 70° deviation
                                          from 180°, so min angle = 180-70 = 110°.
                                          We default to 150° for stricter detection.)
  - |resid(D) - resid(A)| >= HBOND_RES_DIFF when same chain  (default 1)

Classifications match getcontacts:
  hbbb, hbss, hbsb, hbls, hblb, hbll
"""

from __future__ import annotations
import numpy as np
import jax.numpy as jnp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..geometry import fused_hbond_mask


def precompute_hbond_mask(
    groups: ChemicalGroups,
    res_diff: int,
) -> np.ndarray:
    """
    Returns (D, A) bool — statically valid donor-acceptor pairs.
    """
    D = len(groups.donor_indices)
    A = len(groups.acceptor_indices)
    if D == 0 or A == 0:
        return np.empty((0, 0), dtype=bool)

    sele_mask = build_dual_sele_mask(
        groups.donor_in_sele1, groups.donor_in_sele2,
        groups.acceptor_in_sele1, groups.acceptor_in_sele2,
    )
    res_mask = build_residue_diff_mask(
        groups.donor_chain, groups.donor_resid,
        groups.acceptor_chain, groups.acceptor_resid,
        min_diff=res_diff,
        bb_a=groups.donor_is_bb,
        bb_b=groups.acceptor_is_bb,
    )

    return sele_mask & res_mask


def _classify_hbond(d_bb: bool, d_lig: bool, a_bb: bool, a_lig: bool) -> str:
    if d_lig and a_lig:
        return "hbll"
    if d_lig:
        return "hblb" if a_bb else "hbls"
    if a_lig:
        return "hblb" if d_bb else "hbls"
    if d_bb and a_bb:
        return "hbbb"
    if not d_bb and not a_bb:
        return "hbss"
    return "hbsb"


def compute_hbonds(
    coords: jnp.ndarray,       # (F, N, 3)
    groups: ChemicalGroups,
    geom: dict,
    frame_offset: int,
    pair_mask: np.ndarray,     # (D, A) from precompute_hbond_mask
) -> list[tuple]:

    dist_cutoff = float(geom.get("HBOND_CUTOFF_DIST", 3.5))
    ang_cutoff = float(geom.get("HBOND_CUTOFF_ANG", 150.0))

    D = len(groups.donor_indices)
    A = len(groups.acceptor_indices)
    if pair_mask.size == 0 or D == 0 or A == 0:
        return []

    # Gather coordinates
    d_xyz = coords[:, groups.donor_indices, :]     # (F, D, 3)
    h_xyz = coords[:, groups.hydrogen_indices, :]  # (F, D, 3)
    a_xyz = coords[:, groups.acceptor_indices, :]  # (F, A, 3)

    bool_mask = np.array(fused_hbond_mask(
        d_xyz, h_xyz, a_xyz,
        dist_cutoff**2, ang_cutoff
    ))  # (F, D, A)
    
    valid = bool_mask & pair_mask[None, :, :]

    contacts = []
    frames, d_pos, a_pos = np.nonzero(valid)
    for f, di, ai in zip(frames, d_pos, a_pos):
        itype = _classify_hbond(
            bool(groups.donor_is_bb[di]),
            bool(groups.donor_is_ligand[di]),
            bool(groups.acceptor_is_bb[ai]),
            bool(groups.acceptor_is_ligand[ai]),
        )
        contacts.append((
            frame_offset + int(f), itype,
            groups.donor_labels[di], groups.acceptor_labels[ai],
        ))
    return contacts
