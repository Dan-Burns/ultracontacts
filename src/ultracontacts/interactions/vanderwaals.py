"""
vanderwaals.py — GPU-fused van der Waals contact detection via CuPy.

Criterion: distance(a1, a2) < vdw_radius1 + vdw_radius2 + VDW_EPSILON.
Per-pair cutoffs computed inline in the CUDA kernel from radius arrays.
No sub-blocking needed — the kernel never materializes an N² tensor.
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..kernels import vdw_contacts_gpu


def precompute_vdw_mask(groups: ChemicalGroups, res_diff: int,
                         sele1_eq_sele2: bool) -> tuple[np.ndarray, np.ndarray]:
    V1, V2 = len(groups.vdw_indices1), len(groups.vdw_indices2)
    if V1 == 0 or V2 == 0:
        return np.empty((0, 0), dtype=bool), np.empty(0, dtype=np.float32)

    sele_mask = np.ones((V1, V2), dtype=bool)
    res_mask = build_residue_diff_mask(
        groups.vdw_chain1, groups.vdw_resid1,
        groups.vdw_chain2, groups.vdw_resid2,
        min_diff=res_diff,
    )
    pair_mask = sele_mask & res_mask
    if sele1_eq_sele2:
        pair_mask &= np.triu(np.ones_like(pair_mask), k=1)

    return pair_mask, groups.vdw_radii1  # radii2 accessed from groups directly


def precompute_vdw_gpu(groups: ChemicalGroups, pair_mask: np.ndarray) -> dict:
    return {
        "idx1": cp.asarray(groups.vdw_indices1),
        "idx2": cp.asarray(groups.vdw_indices2),
        "mask": cp.asarray(pair_mask.ravel()),
        "radii1": cp.asarray(groups.vdw_radii1),
        "radii2": cp.asarray(groups.vdw_radii2),
        "lbls1": np.array(groups.vdw_labels1, dtype=object),
        "lbls2": np.array(groups.vdw_labels2, dtype=object),
    }


def compute_vanderwaals(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epsilon = float(geom.get("VDW_EPSILON", 0.5))

    if len(gpu_data["idx1"]) == 0 or len(gpu_data["idx2"]) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    i_hits, j_hits = vdw_contacts_gpu(
        coords_gpu, gpu_data["idx1"], gpu_data["idx2"],
        gpu_data["mask"], gpu_data["radii1"], gpu_data["radii2"],
        epsilon,
    )
    if len(i_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(i_hits), abs_frame, dtype=np.int32)
    itypes = np.full(len(i_hits), "vdw", dtype=object)
    a1 = gpu_data["lbls1"][i_hits]
    a2 = gpu_data["lbls2"][j_hits]

    return frames, itypes, a1, a2
