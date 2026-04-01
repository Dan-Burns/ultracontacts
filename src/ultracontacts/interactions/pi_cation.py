"""
pi_cation.py — GPU-fused pi-cation interaction detection via CuPy.

Criterion:
  - cation to ring centroid distance < PC_CUTOFF_DIST (default 6.0 Å)
  - angle between ring normal and centroid→cation vector < PC_CUTOFF_ANG (default 60°)
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..kernels import pi_cation_gpu


def precompute_pc_mask(groups: ChemicalGroups) -> np.ndarray:
    R = len(groups.ring_indices)
    Nc = len(groups.cation_indices)
    if R == 0 or Nc == 0:
        return np.empty((0, 0), dtype=bool)
    return build_dual_sele_mask(
        groups.ring_in_sele1, groups.ring_in_sele2,
        groups.cation_in_sele1, groups.cation_in_sele2,
    )


def precompute_pc_gpu(groups: ChemicalGroups, mask: np.ndarray) -> dict:
    return {
        "ring_atoms": cp.asarray(groups.ring_indices.ravel().astype(np.int32)),
        "cation_idx": cp.asarray(groups.cation_indices),
        "mask": cp.asarray(mask.ravel()),
        "R": len(groups.ring_indices),
        "r_lbls": np.array(groups.ring_labels, dtype=object),
        "c_lbls": np.array(groups.cation_labels, dtype=object),
    }


def compute_pi_cation(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist_cutoff = float(geom.get("PC_CUTOFF_DIST", 6.0))
    ang_cutoff = float(geom.get("PC_CUTOFF_ANG", 60.0))

    R = gpu_data["R"]
    if R == 0 or len(gpu_data["cation_idx"]) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    r_hits, c_hits = pi_cation_gpu(
        coords_gpu, gpu_data["ring_atoms"], gpu_data["cation_idx"],
        gpu_data["mask"], dist_cutoff ** 2, ang_cutoff, R=R,
    )
    if len(r_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(r_hits), abs_frame, dtype=np.int32)
    itypes = np.full(len(r_hits), "pc", dtype=object)
    c_lbls = gpu_data["c_lbls"][c_hits]
    r_lbls = gpu_data["r_lbls"][r_hits]

    return frames, itypes, c_lbls, r_lbls
