"""
salt_bridges.py — GPU-fused salt bridge detection via CuPy.

Criterion: distance between anion atom and cation atom < SB_CUTOFF_DIST (default 4.0 Å).
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..kernels import dist_contacts_gpu


def precompute_sb_mask(groups: ChemicalGroups, sele1_eq_sele2: bool) -> np.ndarray:
    if len(groups.anion_indices) == 0 or len(groups.cation_indices) == 0:
        return np.empty((0, 0), dtype=bool)
    return build_dual_sele_mask(
        groups.anion_in_sele1, groups.anion_in_sele2,
        groups.cation_in_sele1, groups.cation_in_sele2,
    )


def precompute_sb_gpu(groups: ChemicalGroups, mask: np.ndarray) -> dict:
    """Upload static data to GPU once."""
    return {
        "idx1": cp.asarray(groups.anion_indices),
        "idx2": cp.asarray(groups.cation_indices),
        "mask": cp.asarray(mask.ravel()),
        "a_lbls": np.array(groups.anion_labels, dtype=object),
        "c_lbls": np.array(groups.cation_labels, dtype=object),
        "a_lig": np.array(groups.anion_is_ligand, dtype=bool),
        "c_lig": np.array(groups.cation_is_ligand, dtype=bool),
    }


def compute_salt_bridges(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cutoff = float(geom.get("SB_CUTOFF_DIST", 4.0))

    if len(gpu_data["idx1"]) == 0 or len(gpu_data["idx2"]) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    i_hits, j_hits = dist_contacts_gpu(
        coords_gpu, gpu_data["idx1"], gpu_data["idx2"],
        gpu_data["mask"], cutoff ** 2,
    )
    if len(i_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(i_hits), abs_frame, dtype=np.int32)
    a_lbls = gpu_data["a_lbls"][i_hits]
    c_lbls = gpu_data["c_lbls"][j_hits]

    itypes = np.full(len(i_hits), "sb", dtype=object)
    a_lig = gpu_data["a_lig"][i_hits]
    c_lig = gpu_data["c_lig"][j_hits]
    itypes[a_lig & ~c_lig] = "sblp"
    itypes[~a_lig & c_lig] = "sbpl"
    itypes[a_lig & c_lig] = "sbll"

    return frames, itypes, a_lbls, c_lbls
