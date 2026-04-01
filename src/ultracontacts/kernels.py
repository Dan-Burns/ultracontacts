"""
kernels.py — CuPy fused CUDA kernels for molecular contact detection.

Each kernel evaluates one candidate pair per CUDA thread, checks geometric
criteria, and writes only the passing (i, j) indices to a sparse output
buffer via atomicAdd.  The key advantage over the previous JAX approach:
the full (F, N1, N2) boolean tensor is NEVER materialized.
"""

from __future__ import annotations
import numpy as np
import cupy as cp

# ───────────────────────────────────────────────────────────────────
# CUDA kernel sources
# ───────────────────────────────────────────────────────────────────

_DIST_CONTACTS_SRC = r'''
extern "C" __global__
void dist_contacts(
    const float* __restrict__ coords,   // (N_atoms * 3,)
    const int*   __restrict__ idx1,     // (V1,)
    const int*   __restrict__ idx2,     // (V2,)
    const bool*  __restrict__ mask,     // (V1 * V2,)
    float cut_sq,
    int*  out_i,
    int*  out_j,
    int*  out_count,
    int V1, int V2, int max_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= V1 * V2) return;
    if (!mask[tid]) return;

    int i = tid / V2;
    int j = tid % V2;
    int ai = idx1[i] * 3;
    int aj = idx2[j] * 3;

    float dx = coords[ai]   - coords[aj];
    float dy = coords[ai+1] - coords[aj+1];
    float dz = coords[ai+2] - coords[aj+2];
    float d2 = dx*dx + dy*dy + dz*dz;

    if (d2 < cut_sq) {
        int pos = atomicAdd(out_count, 1);
        if (pos < max_out) {
            out_i[pos] = i;
            out_j[pos] = j;
        }
    }
}
'''

_VDW_CONTACTS_SRC = r'''
extern "C" __global__
void vdw_contacts(
    const float* __restrict__ coords,
    const int*   __restrict__ idx1,
    const int*   __restrict__ idx2,
    const bool*  __restrict__ mask,
    const float* __restrict__ radii1,   // (V1,)
    const float* __restrict__ radii2,   // (V2,)
    float epsilon,
    int*  out_i,
    int*  out_j,
    int*  out_count,
    int V1, int V2, int max_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= V1 * V2) return;
    if (!mask[tid]) return;

    int i = tid / V2;
    int j = tid % V2;
    int ai = idx1[i] * 3;
    int aj = idx2[j] * 3;

    float dx = coords[ai]   - coords[aj];
    float dy = coords[ai+1] - coords[aj+1];
    float dz = coords[ai+2] - coords[aj+2];
    float d2 = dx*dx + dy*dy + dz*dz;

    float cut = radii1[i] + radii2[j] + epsilon;
    if (d2 < cut * cut) {
        int pos = atomicAdd(out_count, 1);
        if (pos < max_out) {
            out_i[pos] = i;
            out_j[pos] = j;
        }
    }
}
'''

_HBOND_CONTACTS_SRC = r'''
extern "C" __global__
void hbond_contacts(
    const float* __restrict__ coords,
    const int*   __restrict__ donor_idx,  // (D,) heavy-atom indices
    const int*   __restrict__ h_idx,      // (D,) hydrogen indices
    const int*   __restrict__ acc_idx,    // (A,) acceptor indices
    const bool*  __restrict__ mask,       // (D * A,)
    float max_dist_sq,
    float min_angle_deg,
    int*  out_d,
    int*  out_a,
    int*  out_count,
    int D, int A, int max_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= D * A) return;
    if (!mask[tid]) return;

    int d = tid / A;
    int a = tid % A;

    // Donor (D) and Acceptor (A) positions
    int di = donor_idx[d] * 3;
    int ai = acc_idx[a]   * 3;

    float dax = coords[di]   - coords[ai];
    float day = coords[di+1] - coords[ai+1];
    float daz = coords[di+2] - coords[ai+2];
    float da_sq = dax*dax + day*day + daz*daz;
    if (da_sq >= max_dist_sq) return;

    // Hydrogen position
    int hi = h_idx[d] * 3;

    // Angle at H: vectors H→D and H→A
    float hdx = coords[di] - coords[hi];
    float hdy = coords[di+1] - coords[hi+1];
    float hdz = coords[di+2] - coords[hi+2];

    float hax = coords[ai] - coords[hi];
    float hay = coords[ai+1] - coords[hi+1];
    float haz = coords[ai+2] - coords[hi+2];

    float hd_len = sqrtf(hdx*hdx + hdy*hdy + hdz*hdz);
    float ha_len = sqrtf(hax*hax + hay*hay + haz*haz);
    if (hd_len < 1e-6f || ha_len < 1e-6f) return;

    float cos_a = (hdx*hax + hdy*hay + hdz*haz) / (hd_len * ha_len);
    cos_a = fmaxf(-1.0f, fminf(1.0f, cos_a));
    float angle = acosf(cos_a) * (180.0f / 3.14159265358979f);

    if (angle >= min_angle_deg) {
        int pos = atomicAdd(out_count, 1);
        if (pos < max_out) {
            out_d[pos] = d;
            out_a[pos] = a;
        }
    }
}
'''

_RING_STACKING_SRC = r'''
extern "C" __global__
void ring_stacking(
    const float* __restrict__ coords,
    const int*   __restrict__ ring_atoms,  // (R * 3,)  3 atom indices per ring
    const bool*  __restrict__ mask,        // (R * R,)
    float dist_cut_sq,
    float angle_cut_deg,
    float psi_cut_deg,
    int   is_t_stacking,                   // 0 = pi-stacking, 1 = t-stacking
    int*  out_ri,
    int*  out_rj,
    int*  out_count,
    int R, int max_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= R * R) return;
    if (!mask[tid]) return;

    int ri = tid / R;
    int rj = tid % R;

    // Load ring i atoms and compute centroid + normal
    int a0 = ring_atoms[ri*3+0] * 3;
    int a1 = ring_atoms[ri*3+1] * 3;
    int a2 = ring_atoms[ri*3+2] * 3;

    float cx_i = (coords[a0] + coords[a1] + coords[a2]) / 3.0f;
    float cy_i = (coords[a0+1] + coords[a1+1] + coords[a2+1]) / 3.0f;
    float cz_i = (coords[a0+2] + coords[a1+2] + coords[a2+2]) / 3.0f;

    // Edge vectors for normal
    float e1x = coords[a1]   - coords[a0];
    float e1y = coords[a1+1] - coords[a0+1];
    float e1z = coords[a1+2] - coords[a0+2];
    float e2x = coords[a2]   - coords[a0];
    float e2y = coords[a2+1] - coords[a0+1];
    float e2z = coords[a2+2] - coords[a0+2];

    float nx_i = e1y*e2z - e1z*e2y;
    float ny_i = e1z*e2x - e1x*e2z;
    float nz_i = e1x*e2y - e1y*e2x;
    float n_len_i = sqrtf(nx_i*nx_i + ny_i*ny_i + nz_i*nz_i);
    if (n_len_i < 1e-8f) return;
    nx_i /= n_len_i; ny_i /= n_len_i; nz_i /= n_len_i;

    // Load ring j atoms and compute centroid + normal
    int b0 = ring_atoms[rj*3+0] * 3;
    int b1 = ring_atoms[rj*3+1] * 3;
    int b2 = ring_atoms[rj*3+2] * 3;

    float cx_j = (coords[b0] + coords[b1] + coords[b2]) / 3.0f;
    float cy_j = (coords[b0+1] + coords[b1+1] + coords[b2+1]) / 3.0f;
    float cz_j = (coords[b0+2] + coords[b1+2] + coords[b2+2]) / 3.0f;

    float f1x = coords[b1]   - coords[b0];
    float f1y = coords[b1+1] - coords[b0+1];
    float f1z = coords[b1+2] - coords[b0+2];
    float f2x = coords[b2]   - coords[b0];
    float f2y = coords[b2+1] - coords[b0+1];
    float f2z = coords[b2+2] - coords[b0+2];

    float nx_j = f1y*f2z - f1z*f2y;
    float ny_j = f1z*f2x - f1x*f2z;
    float nz_j = f1x*f2y - f1y*f2x;
    float n_len_j = sqrtf(nx_j*nx_j + ny_j*ny_j + nz_j*nz_j);
    if (n_len_j < 1e-8f) return;
    nx_j /= n_len_j; ny_j /= n_len_j; nz_j /= n_len_j;

    // Centroid distance
    float dx = cx_i - cx_j;
    float dy = cy_i - cy_j;
    float dz = cz_i - cz_j;
    float d2 = dx*dx + dy*dy + dz*dz;
    if (d2 >= dist_cut_sq) return;

    // Plane angle: angle between normals (use absolute cos for sign ambiguity)
    float cos_plane = fabsf(nx_i*nx_j + ny_i*ny_j + nz_i*nz_j);
    cos_plane = fminf(1.0f, cos_plane);
    float plane_angle = acosf(cos_plane) * (180.0f / 3.14159265358979f);

    // Pi-stacking: planes nearly parallel → plane_angle < angle_cut (from 0°)
    // T-stacking:  planes nearly perpendicular → |plane_angle - 90| < angle_cut
    if (is_t_stacking == 0) {
        if (plane_angle > angle_cut_deg) return;
    } else {
        if (fabsf(plane_angle - 90.0f) > angle_cut_deg) return;
    }

    // Psi angle: angle between centroid→centroid vector and ring_i normal
    float d_len = sqrtf(d2);
    if (d_len < 1e-8f) return;
    float cos_psi = fabsf(nx_i*dx + ny_i*dy + nz_i*dz) / d_len;
    cos_psi = fminf(1.0f, cos_psi);
    float psi = acosf(cos_psi) * (180.0f / 3.14159265358979f);

    if (is_t_stacking == 0) {
        if (psi > psi_cut_deg) return;
    } else {
        if (fabsf(psi - 90.0f) > psi_cut_deg) return;
    }

    int pos = atomicAdd(out_count, 1);
    if (pos < max_out) {
        out_ri[pos] = ri;
        out_rj[pos] = rj;
    }
}
'''

_PI_CATION_SRC = r'''
extern "C" __global__
void pi_cation(
    const float* __restrict__ coords,
    const int*   __restrict__ ring_atoms,  // (R * 3,)
    const int*   __restrict__ cation_idx,  // (C,)
    const bool*  __restrict__ mask,        // (R * C,)
    float dist_cut_sq,
    float angle_cut_deg,
    int*  out_r,
    int*  out_c,
    int*  out_count,
    int R, int C, int max_out)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= R * C) return;
    if (!mask[tid]) return;

    int r = tid / C;
    int c = tid % C;

    // Ring centroid and normal
    int a0 = ring_atoms[r*3+0] * 3;
    int a1 = ring_atoms[r*3+1] * 3;
    int a2 = ring_atoms[r*3+2] * 3;

    float cx = (coords[a0] + coords[a1] + coords[a2]) / 3.0f;
    float cy = (coords[a0+1] + coords[a1+1] + coords[a2+1]) / 3.0f;
    float cz = (coords[a0+2] + coords[a1+2] + coords[a2+2]) / 3.0f;

    float e1x = coords[a1]   - coords[a0];
    float e1y = coords[a1+1] - coords[a0+1];
    float e1z = coords[a1+2] - coords[a0+2];
    float e2x = coords[a2]   - coords[a0];
    float e2y = coords[a2+1] - coords[a0+1];
    float e2z = coords[a2+2] - coords[a0+2];

    float nx = e1y*e2z - e1z*e2y;
    float ny = e1z*e2x - e1x*e2z;
    float nz = e1x*e2y - e1y*e2x;
    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (n_len < 1e-8f) return;
    nx /= n_len; ny /= n_len; nz /= n_len;

    // Cation position
    int ci = cation_idx[c] * 3;

    // Centroid → cation vector
    float vx = coords[ci]   - cx;
    float vy = coords[ci+1] - cy;
    float vz = coords[ci+2] - cz;
    float v_sq = vx*vx + vy*vy + vz*vz;

    if (v_sq >= dist_cut_sq) return;

    // Angle between ring normal and centroid→cation vector
    float v_len = sqrtf(v_sq);
    if (v_len < 1e-8f) return;
    float cos_a = fabsf(nx*vx + ny*vy + nz*vz) / v_len;
    cos_a = fminf(1.0f, cos_a);
    float angle = acosf(cos_a) * (180.0f / 3.14159265358979f);

    if (angle < angle_cut_deg) {
        int pos = atomicAdd(out_count, 1);
        if (pos < max_out) {
            out_r[pos] = r;
            out_c[pos] = c;
        }
    }
}
'''

# ───────────────────────────────────────────────────────────────────
# Compiled kernel objects (compiled once on first use)
# ───────────────────────────────────────────────────────────────────

_dist_kernel = cp.RawKernel(_DIST_CONTACTS_SRC, 'dist_contacts')
_vdw_kernel  = cp.RawKernel(_VDW_CONTACTS_SRC,  'vdw_contacts')
_hb_kernel   = cp.RawKernel(_HBOND_CONTACTS_SRC, 'hbond_contacts')
_ring_kernel = cp.RawKernel(_RING_STACKING_SRC,  'ring_stacking')
_pc_kernel   = cp.RawKernel(_PI_CATION_SRC,      'pi_cation')

_BLOCK = 256
_DEFAULT_MAX_HITS = 2_000_000

# Persistent GPU output buffers (allocated once, reused every frame)
_out_i = cp.empty(_DEFAULT_MAX_HITS, dtype=cp.int32)
_out_j = cp.empty(_DEFAULT_MAX_HITS, dtype=cp.int32)
_out_count = cp.zeros(1, dtype=cp.int32)


def _launch(kernel, total_threads, args):
    """Launch a kernel with the standard block size."""
    grid = ((total_threads + _BLOCK - 1) // _BLOCK,)
    kernel(grid, (_BLOCK,), args)


def _read_hits():
    """Read counter, return (i_hits, j_hits) as numpy int32 arrays."""
    n = int(_out_count[0])
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    n = min(n, _DEFAULT_MAX_HITS)
    return _out_i[:n].get(), _out_j[:n].get()


def _reset_counter():
    _out_count[0] = 0


# ───────────────────────────────────────────────────────────────────
# Public Python wrappers
# ───────────────────────────────────────────────────────────────────

def dist_contacts_gpu(
    coords_gpu: cp.ndarray,     # (N_atoms * 3,) float32 on GPU
    idx1_gpu: cp.ndarray,       # (V1,) int32 on GPU
    idx2_gpu: cp.ndarray,       # (V2,) int32 on GPU
    mask_gpu: cp.ndarray,       # (V1 * V2,) bool on GPU
    cut_sq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-only contacts (salt bridges, hydrophobics)."""
    V1, V2 = len(idx1_gpu), len(idx2_gpu)
    _reset_counter()
    _launch(_dist_kernel, V1 * V2, (
        coords_gpu, idx1_gpu, idx2_gpu, mask_gpu,
        np.float32(cut_sq),
        _out_i, _out_j, _out_count,
        np.int32(V1), np.int32(V2), np.int32(_DEFAULT_MAX_HITS),
    ))
    return _read_hits()


def vdw_contacts_gpu(
    coords_gpu: cp.ndarray,
    idx1_gpu: cp.ndarray,
    idx2_gpu: cp.ndarray,
    mask_gpu: cp.ndarray,
    radii1_gpu: cp.ndarray,     # (V1,) float32
    radii2_gpu: cp.ndarray,     # (V2,) float32
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Van der Waals contacts with per-pair cutoffs."""
    V1, V2 = len(idx1_gpu), len(idx2_gpu)
    _reset_counter()
    _launch(_vdw_kernel, V1 * V2, (
        coords_gpu, idx1_gpu, idx2_gpu, mask_gpu,
        radii1_gpu, radii2_gpu,
        np.float32(epsilon),
        _out_i, _out_j, _out_count,
        np.int32(V1), np.int32(V2), np.int32(_DEFAULT_MAX_HITS),
    ))
    return _read_hits()


def hbond_contacts_gpu(
    coords_gpu: cp.ndarray,
    donor_idx_gpu: cp.ndarray,  # (D,) int32
    h_idx_gpu: cp.ndarray,      # (D,) int32
    acc_idx_gpu: cp.ndarray,    # (A,) int32
    mask_gpu: cp.ndarray,       # (D * A,) bool
    max_dist_sq: float,
    min_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """H-bond contacts with distance + angle criteria."""
    D, A = len(donor_idx_gpu), len(acc_idx_gpu)
    _reset_counter()
    _launch(_hb_kernel, D * A, (
        coords_gpu, donor_idx_gpu, h_idx_gpu, acc_idx_gpu, mask_gpu,
        np.float32(max_dist_sq), np.float32(min_angle_deg),
        _out_i, _out_j, _out_count,
        np.int32(D), np.int32(A), np.int32(_DEFAULT_MAX_HITS),
    ))
    return _read_hits()


def ring_stacking_gpu(
    coords_gpu: cp.ndarray,
    ring_atoms_gpu: cp.ndarray,  # (R * 3,) int32  [flattened ring_indices]
    mask_gpu: cp.ndarray,        # (R * R,) bool
    dist_cut_sq: float,
    angle_cut_deg: float,
    psi_cut_deg: float,
    is_t_stacking: bool,
    R: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pi-stacking or T-stacking contacts."""
    _reset_counter()
    _launch(_ring_kernel, R * R, (
        coords_gpu, ring_atoms_gpu, mask_gpu,
        np.float32(dist_cut_sq), np.float32(angle_cut_deg), np.float32(psi_cut_deg),
        np.int32(1 if is_t_stacking else 0),
        _out_i, _out_j, _out_count,
        np.int32(R), np.int32(_DEFAULT_MAX_HITS),
    ))
    return _read_hits()


def pi_cation_gpu(
    coords_gpu: cp.ndarray,
    ring_atoms_gpu: cp.ndarray,  # (R * 3,) int32
    cation_idx_gpu: cp.ndarray,  # (C,) int32
    mask_gpu: cp.ndarray,        # (R * C,) bool
    dist_cut_sq: float,
    angle_cut_deg: float,
    R: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pi-cation contacts."""
    C = len(cation_idx_gpu)
    _reset_counter()
    _launch(_pc_kernel, R * C, (
        coords_gpu, ring_atoms_gpu, cation_idx_gpu, mask_gpu,
        np.float32(dist_cut_sq), np.float32(angle_cut_deg),
        _out_i, _out_j, _out_count,
        np.int32(R), np.int32(C), np.int32(_DEFAULT_MAX_HITS),
    ))
    return _read_hits()
