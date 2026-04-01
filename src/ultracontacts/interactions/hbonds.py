"""
hbonds.py — GPU-fused hydrogen bond detection via CuPy.

Criteria:
  - D···A distance < HBOND_CUTOFF_DIST (default 3.5 Å)
  - D-H···A angle  > HBOND_CUTOFF_ANG  (default 150°)
  - |resid(D) - resid(A)| >= HBOND_RES_DIFF
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..kernels import hbond_contacts_gpu


def precompute_hbond_mask(groups: ChemicalGroups, res_diff: int) -> np.ndarray:
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


def precompute_hb_gpu(groups: ChemicalGroups, mask: np.ndarray) -> dict:
    return {
        "donor_idx": cp.asarray(groups.donor_indices),
        "h_idx": cp.asarray(groups.hydrogen_indices),
        "acc_idx": cp.asarray(groups.acceptor_indices),
        "mask": cp.asarray(mask.ravel()),
        "d_lbls": np.array(groups.donor_labels, dtype=object),
        "a_lbls": np.array(groups.acceptor_labels, dtype=object),
        "d_lig": np.array(groups.donor_is_ligand, dtype=bool),
        "a_lig": np.array(groups.acceptor_is_ligand, dtype=bool),
    }


def compute_hbonds(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist_cutoff = float(geom.get("HBOND_CUTOFF_DIST", 3.5))
    ang_cutoff = float(geom.get("HBOND_CUTOFF_ANG", 150.0))

    if len(gpu_data["donor_idx"]) == 0 or len(gpu_data["acc_idx"]) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    d_hits, a_hits = hbond_contacts_gpu(
        coords_gpu, gpu_data["donor_idx"], gpu_data["h_idx"], gpu_data["acc_idx"],
        gpu_data["mask"], dist_cutoff ** 2, ang_cutoff,
    )
    if len(d_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(d_hits), abs_frame, dtype=np.int32)
    d_lbls = gpu_data["d_lbls"][d_hits]
    a_lbls = gpu_data["a_lbls"][a_hits]

    d_lig = gpu_data["d_lig"][d_hits]
    a_lig = gpu_data["a_lig"][a_hits]
    itypes = np.full(len(d_hits), "hb", dtype=object)
    itypes[d_lig & ~a_lig] = "hblp"
    itypes[~d_lig & a_lig] = "hbpl"
    itypes[d_lig & a_lig] = "hbll"

    return frames, itypes, d_lbls, a_lbls
