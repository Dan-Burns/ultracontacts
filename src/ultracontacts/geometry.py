"""
geometry.py — JAX-JIT-compiled geometric primitives.

All functions operate on batched arrays and run on GPU.
Utilizes FUSED bitmask kernels AND Batched GEMM cuBLAS multiplications 
to ensure minimum VRAM usage and absolute peak CUDA SM saturation.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Core Aromatic Plane Helpers
# ---------------------------------------------------------------------------

@jax.jit
def ring_centroids(ring_xyz: jnp.ndarray) -> jnp.ndarray:
    """ ring_xyz: (F, ..., 3, 3) -> returns: (F, ..., 3) """
    return jnp.mean(ring_xyz, axis=-2)

@jax.jit
def ring_normals(ring_xyz: jnp.ndarray) -> jnp.ndarray:
    """ ring_xyz: (F, ..., 3, 3) -> returns: (F, ..., 3) """
    v1 = ring_xyz[..., 1, :] - ring_xyz[..., 0, :]
    v2 = ring_xyz[..., 2, :] - ring_xyz[..., 0, :]
    n = jnp.cross(v1, v2)
    norm = jnp.linalg.norm(n, axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    return n / norm

# ---------------------------------------------------------------------------
# GEMM FUSED KERNELS
# ---------------------------------------------------------------------------

@jax.jit
def fused_dist_mask(a: jnp.ndarray, b: jnp.ndarray, max_d_sq: jnp.ndarray | float) -> jnp.ndarray:
    """
    Computes batched Euclidean distance using cuBLAS Matmul:
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2(A*B^T)
    """
    a_sq = jnp.sum(a * a, axis=-1)[..., :, None]
    b_sq = jnp.sum(b * b, axis=-1)[..., None, :]
    ab = jnp.matmul(a, jnp.swapaxes(b, -1, -2))
    dists_sq = jnp.clip(a_sq + b_sq - 2.0 * ab, 0.0, None)
    return dists_sq < max_d_sq


@jax.jit
def fused_hbond_mask(
    donor_xyz: jnp.ndarray,
    hydrogen_xyz: jnp.ndarray,
    acceptor_xyz: jnp.ndarray,
    max_d_sq: float,
    min_angle: float,
) -> jnp.ndarray:
    """
    GEMM implementation of Hydrogen Bonds without (F, D, A, 3) diff arrays.
    """
    # 1. Distance D..A using Matmul
    d_sq = jnp.sum(donor_xyz * donor_xyz, axis=-1)[..., :, None]
    a_sq = jnp.sum(acceptor_xyz * acceptor_xyz, axis=-1)[..., None, :]
    da_dot = jnp.matmul(donor_xyz, jnp.swapaxes(acceptor_xyz, -1, -2))
    dists_sq = jnp.clip(d_sq + a_sq - 2.0 * da_dot, 0.0, None)
    
    # 2. Angle calculation
    dh = hydrogen_xyz - donor_xyz                                     # (F, D, 3)
    dh_norm_sq = jnp.sum(dh * dh, axis=-1)[..., :, None]              # (F, D, 1)
    dh_norm = jnp.sqrt(dh_norm_sq)
    dh_norm = jnp.where(dh_norm < 1e-8, 1.0, dh_norm)
    
    # Needs ha_norm for angle, distance between H and A
    h_sq = jnp.sum(hydrogen_xyz * hydrogen_xyz, axis=-1)[..., :, None]
    ha_dot = jnp.matmul(hydrogen_xyz, jnp.swapaxes(acceptor_xyz, -1, -2))
    ha_dists_sq = jnp.clip(h_sq + a_sq - 2.0 * ha_dot, 0.0, None)
    
    ha_norm = jnp.sqrt(ha_dists_sq)                                   # (F, D, A)
    ha_norm = jnp.where(ha_norm < 1e-8, 1.0, ha_norm)
    
    # dh \cdot (a - h) = dh \cdot a  -  dh \cdot h
    dh_dot_a = jnp.matmul(dh, jnp.swapaxes(acceptor_xyz, -1, -2))     # (F, D, A)
    dh_dot_h = jnp.sum(dh * hydrogen_xyz, axis=-1)[..., :, None]      # (F, D, 1)
    cos_angle_num = dh_dot_a - dh_dot_h                               # (F, D, A)
    
    cos_angle = cos_angle_num / (dh_norm * ha_norm)
    angles = jnp.degrees(jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0)))
    
    return (dists_sq < max_d_sq) & (angles >= min_angle)


@jax.jit
def fused_pi_stacking_mask(
    ring_xyz: jnp.ndarray,
    dist_cut_sq: float,
    ang_cut: float,
    psi_cut: float,
) -> jnp.ndarray:
    cents = ring_centroids(ring_xyz)  # (F, R, 3)
    norms = ring_normals(ring_xyz)    # (F, R, 3)
    
    # 1. Centroid Distances via Matmul
    c_sq = jnp.sum(cents * cents, axis=-1)
    cc_dot = jnp.matmul(cents, jnp.swapaxes(cents, -1, -2))
    cent_dists_sq = jnp.clip(c_sq[..., :, None] + c_sq[..., None, :] - 2.0 * cc_dot, 0.0, None)
    
    # 2. Plane Angle via Matmul
    n_dot = jnp.matmul(norms, jnp.swapaxes(norms, -1, -2))
    cos_angle = jnp.clip(jnp.abs(n_dot), 0.0, 1.0)
    plane_angle = jnp.degrees(jnp.arccos(cos_angle))
    
    # 3. Psi Angles via Matmul
    vec_norm = jnp.sqrt(cent_dists_sq)
    vec_norm = jnp.where(vec_norm < 1e-8, 1.0, vec_norm)
    
    c_dot_n = jnp.matmul(norms, jnp.swapaxes(cents, -1, -2))             # (F, R_norm, R_cent)
    c_dot_n_self = jnp.sum(norms * cents, axis=-1)[..., :, None]         # (F, R_n_c, 1)
    psi1_cos = (c_dot_n - c_dot_n_self) / vec_norm
    psi1 = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(psi1_cos), 0.0, 1.0)))
    
    n_dot_c = jnp.matmul(cents, jnp.swapaxes(norms, -1, -2))             # (F, R_cent, R_norm)
    c_dot_n_self_2 = jnp.sum(cents * norms, axis=-1)[..., None, :]       # (F, 1, R_n_c)
    psi2_cos = (c_dot_n_self_2 - n_dot_c) / vec_norm
    psi2 = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(psi2_cos), 0.0, 1.0)))
    
    psi = jnp.minimum(psi1, psi2)
    ang_ok = (plane_angle < ang_cut)
    return (cent_dists_sq < dist_cut_sq) & ang_ok & (psi < psi_cut)


@jax.jit
def fused_t_stacking_mask(
    ring_xyz: jnp.ndarray,
    dist_cut_sq: float,
    ang_cut: float,
    psi_cut: float,
) -> jnp.ndarray:
    cents = ring_centroids(ring_xyz)  
    norms = ring_normals(ring_xyz)    
    
    c_sq = jnp.sum(cents * cents, axis=-1)
    cc_dot = jnp.matmul(cents, jnp.swapaxes(cents, -1, -2))
    cent_dists_sq = jnp.clip(c_sq[..., :, None] + c_sq[..., None, :] - 2.0 * cc_dot, 0.0, None)
    
    n_dot = jnp.matmul(norms, jnp.swapaxes(norms, -1, -2))
    cos_angle = jnp.clip(jnp.abs(n_dot), 0.0, 1.0)
    plane_angle = jnp.degrees(jnp.arccos(cos_angle))
    
    vec_norm = jnp.sqrt(cent_dists_sq)
    vec_norm = jnp.where(vec_norm < 1e-8, 1.0, vec_norm)
    
    c_dot_n = jnp.matmul(norms, jnp.swapaxes(cents, -1, -2))             
    c_dot_n_self = jnp.sum(norms * cents, axis=-1)[..., :, None]         
    psi1_cos = (c_dot_n - c_dot_n_self) / vec_norm
    psi1 = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(psi1_cos), 0.0, 1.0)))
    
    n_dot_c = jnp.matmul(cents, jnp.swapaxes(norms, -1, -2))             
    c_dot_n_self_2 = jnp.sum(cents * norms, axis=-1)[..., None, :]       
    psi2_cos = (c_dot_n_self_2 - n_dot_c) / vec_norm
    psi2 = jnp.degrees(jnp.arccos(jnp.clip(jnp.abs(psi2_cos), 0.0, 1.0)))
    
    psi = jnp.minimum(psi1, psi2)
    ang_ok = (jnp.abs(plane_angle - 90.0) < ang_cut)
    return (cent_dists_sq < dist_cut_sq) & ang_ok & (psi < psi_cut)


@jax.jit
def fused_pi_cation_mask(
    ring_xyz: jnp.ndarray,
    cation_xyz: jnp.ndarray,
    dist_cut_sq: float,
    offset_cut: float,
    ang_cut: float,
) -> jnp.ndarray:
    cents = ring_centroids(ring_xyz)  # (F, R, 3)
    norms = ring_normals(ring_xyz)    # (F, R, 3)
    
    c_sq = jnp.sum(cents * cents, axis=-1)[..., :, None]
    cat_sq = jnp.sum(cation_xyz * cation_xyz, axis=-1)[..., None, :]
    ccat_dot = jnp.matmul(cents, jnp.swapaxes(cation_xyz, -1, -2))
    dists_sq = jnp.clip(c_sq + cat_sq - 2.0 * ccat_dot, 0.0, None)
    
    cat_dot_n = jnp.matmul(norms, jnp.swapaxes(cation_xyz, -1, -2)) # (F, R, C)
    cent_dot_n = jnp.sum(cents * norms, axis=-1)[..., :, None]      # (F, R, 1)
    proj = jnp.abs(cat_dot_n - cent_dot_n)                          # (F, R, C)
    offset_sq = dists_sq - proj * proj
    
    vec_norm = jnp.sqrt(dists_sq)
    vec_norm = jnp.where(vec_norm < 1e-8, 1.0, vec_norm)
    dot = jnp.clip(proj / vec_norm, 0.0, 1.0)
    angle = jnp.degrees(jnp.arccos(dot))
    
    return (dists_sq < dist_cut_sq) & (offset_sq < offset_cut**2) & (angle < ang_cut)
