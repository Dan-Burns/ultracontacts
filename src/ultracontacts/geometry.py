"""
geometry.py — JAX-JIT-compiled geometric primitives.

All functions operate on batched arrays and run on GPU.
No topology logic here — pure math.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------------

@jax.jit
def pairwise_dist_sq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Squared pairwise distances.

    a: (Na, 3)
    b: (Nb, 3)
    returns: (Na, Nb)
    """
    diff = a[:, None, :] - b[None, :, :]   # (Na, Nb, 3)
    return jnp.sum(diff * diff, axis=-1)


@jax.jit
def batch_pairwise_dist_sq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Batched squared pairwise distances.

    a: (F, Na, 3)
    b: (F, Nb, 3)
    returns: (F, Na, Nb)
    """
    diff = a[:, :, None, :] - b[:, None, :, :]  # (F, Na, Nb, 3)
    return jnp.sum(diff * diff, axis=-1)


# ---------------------------------------------------------------------------
# Aromatic ring geometry
# ---------------------------------------------------------------------------

@jax.jit
def ring_centroids(ring_xyz: jnp.ndarray) -> jnp.ndarray:
    """
    ring_xyz: (F, R, 3, 3)  — F frames, R rings, 3 atoms, xyz
    returns:  (F, R, 3)
    """
    return jnp.mean(ring_xyz, axis=2)


@jax.jit
def ring_normals(ring_xyz: jnp.ndarray) -> jnp.ndarray:
    """
    Normalized normal vectors of aromatic planes by cross product.

    ring_xyz: (F, R, 3, 3)
    returns:  (F, R, 3)
    """
    v1 = ring_xyz[:, :, 1, :] - ring_xyz[:, :, 0, :]  # (F, R, 3)
    v2 = ring_xyz[:, :, 2, :] - ring_xyz[:, :, 0, :]  # (F, R, 3)
    n = jnp.cross(v1, v2)                              # (F, R, 3)
    norm = jnp.linalg.norm(n, axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    return n / norm


@jax.jit
def angle_between_normals_deg(n1: jnp.ndarray, n2: jnp.ndarray) -> jnp.ndarray:
    """
    Angle in degrees between pairs of unit vectors (takes absolute of dot product
    to handle anti-parallel normals as equivalent).

    n1, n2: (..., 3)
    returns: (...)
    """
    dot = jnp.clip(jnp.abs(jnp.sum(n1 * n2, axis=-1)), 0.0, 1.0)
    return jnp.degrees(jnp.arccos(dot))


@jax.jit
def psi_angle_deg(c1: jnp.ndarray, c2: jnp.ndarray, normal: jnp.ndarray) -> jnp.ndarray:
    """
    Psi angle: angle between (c2-c1) vector and ring normal.
    Measures how 'offset' the stack is.

    c1, c2, normal: (..., 3)
    returns: (...)
    """
    vec = c2 - c1
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    vec_unit = vec / norm
    dot = jnp.clip(jnp.abs(jnp.sum(vec_unit * normal, axis=-1)), 0.0, 1.0)
    return jnp.degrees(jnp.arccos(dot))


# ---------------------------------------------------------------------------
# H-bond angles
# ---------------------------------------------------------------------------

@jax.jit
def dha_angle_deg(
    donor_xyz: jnp.ndarray,
    hydrogen_xyz: jnp.ndarray,
    acceptor_xyz: jnp.ndarray,
) -> jnp.ndarray:
    """
    D-H···A angle in degrees.

    donor_xyz:    (F, D, 3)
    hydrogen_xyz: (F, D, 3)
    acceptor_xyz: (F, A, 3)
    returns:      (F, D, A)
    """
    dh = hydrogen_xyz - donor_xyz            # (F, D, 3)
    # H→A vectors for all (D, A) pairs
    ha = acceptor_xyz[:, None, :, :] - hydrogen_xyz[:, :, None, :]  # (F, D, A, 3)
    dh_exp = dh[:, :, None, :]              # (F, D, 1, 3)

    dh_norm = jnp.linalg.norm(dh_exp, axis=-1, keepdims=True)
    ha_norm = jnp.linalg.norm(ha,     axis=-1, keepdims=True)
    dh_norm = jnp.where(dh_norm < 1e-8, 1.0, dh_norm)
    ha_norm = jnp.where(ha_norm < 1e-8, 1.0, ha_norm)

    cos_angle = jnp.sum((dh_exp / dh_norm) * (ha / ha_norm), axis=-1)  # (F, D, A)
    return jnp.degrees(jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Pi-cation
# ---------------------------------------------------------------------------

@jax.jit
def centroid_to_atom_angle_deg(
    centroids: jnp.ndarray,
    normals: jnp.ndarray,
    cation_xyz: jnp.ndarray,
) -> jnp.ndarray:
    """
    Angle between ring normal and centroid→cation vector.

    centroids:  (F, R, 3)
    normals:    (F, R, 3)
    cation_xyz: (F, C, 3)
    returns:    (F, R, C)
    """
    # centroid→cation: (F, R, C, 3)
    vec = cation_xyz[:, None, :, :] - centroids[:, :, None, :]
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    norm = jnp.where(norm < 1e-8, 1.0, norm)
    vec_unit = vec / norm                                    # (F, R, C, 3)
    n_exp = normals[:, :, None, :]                          # (F, R, 1, 3)
    dot = jnp.clip(jnp.abs(jnp.sum(n_exp * vec_unit, axis=-1)), 0.0, 1.0)  # (F, R, C)
    return jnp.degrees(jnp.arccos(dot))


# ---------------------------------------------------------------------------
# Distance between centroid and a set of atoms
# ---------------------------------------------------------------------------

@jax.jit
def centroid_to_atoms_dist_sq(
    centroids: jnp.ndarray,
    atoms_xyz: jnp.ndarray,
) -> jnp.ndarray:
    """
    centroids:  (F, R, 3)
    atoms_xyz:  (F, A, 3)
    returns:    (F, R, A) squared distances
    """
    diff = centroids[:, :, None, :] - atoms_xyz[:, None, :, :]  # (F, R, A, 3)
    return jnp.sum(diff * diff, axis=-1)
