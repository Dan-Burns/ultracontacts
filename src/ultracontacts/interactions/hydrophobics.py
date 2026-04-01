"""
hydrophobics.py — GPU-fused hydrophobic contact detection via CuPy.

Criterion: two hydrophobic C/S atoms within HP_CUTOFF_DIST (default 4.0 Å),
           in different residues.
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask, build_residue_diff_mask
from ..kernels import dist_contacts_gpu


def precompute_hp_mask(groups: ChemicalGroups, res_diff: int) -> np.ndarray:
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
    np.fill_diagonal(res_mask, False)
    upper = np.triu(np.ones((Hp, Hp), dtype=bool), k=1)
    return sele_mask & res_mask & upper


def precompute_hp_gpu(groups: ChemicalGroups, mask: np.ndarray) -> dict:
    return {
        "idx": cp.asarray(groups.hp_indices),
        "mask": cp.asarray(mask.ravel()),
        "lbls": np.array(groups.hp_labels, dtype=object),
    }


def compute_hydrophobics(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cutoff = float(geom.get("HP_CUTOFF_DIST", 4.0))

    if len(gpu_data["idx"]) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    i_hits, j_hits = dist_contacts_gpu(
        coords_gpu, gpu_data["idx"], gpu_data["idx"],
        gpu_data["mask"], cutoff ** 2,
    )
    if len(i_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(i_hits), abs_frame, dtype=np.int32)
    itypes = np.full(len(i_hits), "hp", dtype=object)
    a1 = gpu_data["lbls"][i_hits]
    a2 = gpu_data["lbls"][j_hits]

    return frames, itypes, a1, a2
