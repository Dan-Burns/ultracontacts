"""
aromatics.py — GPU-fused pi-stacking and T-stacking detection via CuPy.

Pi-stacking  (ps): planes nearly parallel  → normal angle ≈ 0°
T-stacking   (ts): planes nearly perpendicular → normal angle ≈ 90°
"""

from __future__ import annotations
import numpy as np
import cupy as cp

from ..topology import ChemicalGroups, build_dual_sele_mask
from ..kernels import ring_stacking_gpu


def precompute_ring_pair_mask(groups: ChemicalGroups, sele1_eq_sele2: bool) -> np.ndarray:
    R = len(groups.ring_indices)
    if R == 0:
        return np.empty((0, 0), dtype=bool)
    sele_mask = build_dual_sele_mask(
        groups.ring_in_sele1, groups.ring_in_sele2,
        groups.ring_in_sele1, groups.ring_in_sele2,
    )
    np.fill_diagonal(sele_mask, False)
    if sele1_eq_sele2:
        sele_mask &= np.triu(np.ones((R, R), dtype=bool), k=1)
    return sele_mask


def precompute_ring_gpu(groups: ChemicalGroups, mask: np.ndarray) -> dict:
    return {
        "ring_atoms": cp.asarray(groups.ring_indices.ravel().astype(np.int32)),
        "mask": cp.asarray(mask.ravel()),
        "R": len(groups.ring_indices),
        "lbls": np.array(groups.ring_labels, dtype=object),
    }


def _compute_stacking(
    coords_gpu: cp.ndarray,
    gpu_data: dict,
    geom: dict,
    abs_frame: int,
    itype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if itype == "ps":
        dist_cut = float(geom.get("PS_CUTOFF_DIST", 7.0))
        ang_cut = float(geom.get("PS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("PS_PSI_ANG", 45.0))
    else:
        dist_cut = float(geom.get("TS_CUTOFF_DIST", 5.0))
        ang_cut = float(geom.get("TS_CUTOFF_ANG", 30.0))
        psi_cut = float(geom.get("TS_PSI_ANG", 45.0))

    R = gpu_data["R"]
    if R == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    ri_hits, rj_hits = ring_stacking_gpu(
        coords_gpu, gpu_data["ring_atoms"], gpu_data["mask"],
        dist_cut ** 2, ang_cut, psi_cut,
        is_t_stacking=(itype == "ts"),
        R=R,
    )
    if len(ri_hits) == 0:
        return (np.empty(0, np.int32), np.empty(0, object), np.empty(0, object), np.empty(0, object))

    frames = np.full(len(ri_hits), abs_frame, dtype=np.int32)
    itypes = np.full(len(ri_hits), itype, dtype=object)
    a1 = gpu_data["lbls"][ri_hits]
    a2 = gpu_data["lbls"][rj_hits]

    return frames, itypes, a1, a2


def compute_pi_stacking(coords_gpu, gpu_data, geom, abs_frame):
    return _compute_stacking(coords_gpu, gpu_data, geom, abs_frame, "ps")


def compute_t_stacking(coords_gpu, gpu_data, geom, abs_frame):
    return _compute_stacking(coords_gpu, gpu_data, geom, abs_frame, "ts")
