# ultracontacts

GPU-accelerated molecular contact analysis. Computes hydrogen bonds, salt bridges, pi-stacking, T-stacking, pi-cation, van der Waals, and hydrophobic contacts across MD trajectories, outputting a Parquet file for downstream analysis.

**~150 frames/second** on a modern GPU (tested on a 6354-frame, ~6000-atom protein trajectory completing in ~42 seconds).

## Requirements

- NVIDIA GPU with CUDA support
- [`cupy`](https://cupy.dev/) (for CUDA kernels) — install matching your CUDA version:
  ```bash
  conda install -c conda-forge cupy
  # or
  pip install cupy-cuda12x  # adjust for your CUDA version
  ```
- `MDAnalysis`, `pyarrow`, `tqdm`, `numpy`

## Installation

```bash
pip install -e .
```

## Usage

### Basic

```bash
ultracontacts \
  --topology protein.pdb \
  --trajectory sim.dcd \
  --output contacts.parquet
```

### With trajectory subset and custom interactions

```bash
ultracontacts \
  --topology system.psf \
  --trajectory production.dcd \
  --itypes hb sb ps ts pc vdw hp \
  --beg 0 --end 999 --stride 2 \
  --output contacts.parquet
```

### Two-selection mode (e.g. protein–ligand)

```bash
ultracontacts \
  --topology system.prmtop \
  --trajectory md.nc \
  --sele "protein" \
  --sele2 "resname LIG" \
  --itypes hb sb vdw \
  --output prot_lig_contacts.parquet
```

### Compute contact frequencies from output

```bash
ultracontacts \
  --frequencies contacts.parquet \
  --output contact_frequencies.tsv
```

## Interaction Types

| Flag | Name | Criteria |
|------|------|----------|
| `hb` | Hydrogen bond | D···A < 3.5 Å, D-H···A angle > 150° |
| `sb` | Salt bridge | Anion–cation distance < 4.0 Å |
| `ps` | Pi-stacking | Centroid dist < 7.0 Å, plane angle < 30°, psi < 45° |
| `ts` | T-stacking | Centroid dist < 5.0 Å, plane angle ≈ 90° ± 30°, psi ≈ 90° ± 45° |
| `pc` | Pi-cation | Centroid–cation dist < 6.0 Å, normal–cation angle < 60° |
| `vdw` | Van der Waals | Distance < r₁ + r₂ + 0.5 Å |
| `hp` | Hydrophobic | C/S atoms < 4.0 Å (hydrophobic residues only) |

Use `--itypes all` to compute all types.

## Output Format

Contacts are written to a Parquet file with columns:

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int32 | Trajectory frame index |
| `itype` | string | Interaction type (e.g. `hb`, `sb`) |
| `atom1` | string | `chain:resname:resid:name` |
| `atom2` | string | `chain:resname:resid:name` |

Read with pandas:
```python
import pandas as pd
df = pd.read_parquet("contacts.parquet")
print(df.groupby("itype").size())
```

## Geometric Criteria Overrides

All cutoffs can be overridden at the command line:

```bash
ultracontacts \
  --topology protein.pdb \
  --trajectory sim.dcd \
  --output contacts.parquet \
  --hbond-cutoff-dist 3.2 \
  --hbond-cutoff-ang 140 \
  --vdw-epsilon 0.3 \
  --sb-cutoff-dist 5.0
```

Full list: `ultracontacts --help`

## Architecture

Contacts are computed by five fused CUDA kernels (`kernels.py`) compiled via `cupy.RawKernel`:

- **`dist_contacts`** — simple distance threshold (sb, hp)
- **`vdw_contacts`** — per-pair VDW cutoff from atomic radii (vdw)
- **`hbond_contacts`** — distance + D-H···A angle (hb)
- **`ring_stacking`** — centroid distance + plane angle + psi (ps, ts)
- **`pi_cation`** — centroid distance + normal-cation angle (pc)

Each kernel uses one CUDA thread per candidate pair and writes only passing pairs via `atomicAdd` — the full N² distance matrix is never materialized. Static data (atom indices, topology masks, radii) is uploaded to GPU once at startup; only coordinates (~70 KB/frame) are transferred per frame.
