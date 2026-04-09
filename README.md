# ultracontacts

GPU-accelerated molecular contact analysis. Computes hydrogen bonds, salt bridges, pi-stacking, T-stacking, pi-cation, van der Waals, and hydrophobic contacts across MD trajectories, outputting compressed Parquet files for downstream analysis.

**~150 frames/second** on a modern GPU (tested on a 6354-frame, ~20,000-atom protein trajectory completing in ~42 seconds).

## Requirements

- NVIDIA GPU with CUDA support
- [`cupy`](https://cupy.dev/) matching your CUDA version (see [Installation](#installation))
- `MDAnalysis`, `pyarrow`, `polars`, `tqdm`, `numpy`, `rdkit`

## Installation

CuPy wheels are CUDA-version-specific and cannot be auto-selected by pip.
Install the matching wheel for your CUDA runtime, then install ultracontacts:

```bash
# Option A: install CuPy separately, then ultracontacts
pip install cupy-cuda12x      # CUDA 12.x
pip install ultracontacts

# Option B: use the package extra shorthand
pip install ultracontacts[cuda12x]    # CUDA 12.x
pip install ultracontacts[cuda13x]    # CUDA 13.x
pip install ultracontacts[cuda11x]    # CUDA 11.x
```

For development:

```bash
pip install cupy-cuda12x
pip install -e .
```

### Optional: ParmEd (for AMBER/GROMACS/CHARMM bond topology)

If you want to load bond topology from AMBER `.prmtop`, GROMACS `.top`/`.tpr`,
or CHARMM `.psf` files, install ParmEd:

```bash
pip install parmed
# or
pip install ultracontacts[parmed]
```

ParmEd is **not required** for OpenMM `system.xml` files or for basic usage
with `guess_bonds`.

## Usage

### Contact calculation

```bash
ultracontacts contacts \
  --topology protein.pdb \
  --trajectory sim.dcd \
  --output contacts.parquet
```

By default this also writes `contacts_frequencies.parquet` (per-residue-pair contact frequencies). Use `--no-frequencies` to skip this.

### Bond topology for accurate H-bond detection

Accurate H-bond detection requires knowing which atoms are covalently bonded.
PDB CONECT records are often incomplete, so providing the simulation's bond
topology is strongly recommended. The `--bond-topology` flag auto-detects the
file format (Only tested with OpenMM system xml (4/8/2026)):

```bash
# OpenMM
ultracontacts contacts \
  --topology system.pdb \
  --trajectory sim.dcd \
  --bond-topology system.xml \
  --output contacts.parquet

# AMBER
ultracontacts contacts \
  --topology stripped.pdb \
  --trajectory sim.nc \
  --bond-topology full_system.prmtop \
  --output contacts.parquet

# GROMACS
ultracontacts contacts \
  --topology stripped.pdb \
  --trajectory sim.xtc \
  --bond-topology topol.tpr \
  --output contacts.parquet

# CHARMM
ultracontacts contacts \
  --topology stripped.pdb \
  --trajectory sim.dcd \
  --bond-topology system.psf \
  --output contacts.parquet
```

**Full-system topology files work correctly against solvent-stripped structures.**
Bonds involving atom indices beyond the stripped structure are automatically
filtered out — the same file you used to run the simulation can be passed
directly, even if it contains water and ions that were removed from the
trajectory.

| Format | Extensions | Requires |
|--------|-----------|----------|
| OpenMM | `.xml` | (stdlib only) |
| AMBER | `.prmtop`, `.parm7` | `parmed` |
| GROMACS | `.top`, `.tpr` | `parmed` |
| CHARMM | `.psf` | `parmed` |
| Desmond | `.cms` | `parmed` |

### Trajectory subset, custom interactions, two-selection mode

```bash
ultracontacts contacts \
  --topology system.psf \
  --trajectory production.dcd \
  --itypes hb sb ps ts pc \
  --beg 100 --end 4999 --stride 2 \
  --sele "protein" \
  --sele2 "resname LIG" \
  --output prot_lig_contacts.parquet
```

### Frequency output options

```bash
# Frequencies are written automatically — skip with:
ultracontacts contacts ... --no-frequencies

# Custom frequency file path (.tsv extension → TSV format):
ultracontacts contacts ... --frequencies my_freqs.tsv

# Also write condensed wide-format (one row, columns = "res1-res2" pair names):
ultracontacts contacts ... --condensed
```

### Standalone frequency calculation

```bash
# Long format (itype, res1, res2, frequency, count):
ultracontacts frequencies --input contacts.parquet --output freqs.parquet

# Condensed wide-format (one row, residue pair column names, any-itype probability):
ultracontacts frequencies --input contacts.parquet --output freqs_condensed.parquet --condensed

# TSV output:
ultracontacts frequencies --input contacts.parquet --output freqs.tsv
```

## Interaction Types

| Flag | Name | Default Criteria |
|------|------|-----------------|
| `hb` | Hydrogen bond | D···A < 3.5 Å, D-H···A > 110° (matches VMD/getcontacts) |
| `sb` | Salt bridge | Anion–cation distance < 4.0 Å |
| `ps` | Pi-stacking | Centroid dist < 7.0 Å, plane angle < 30°, psi < 45° |
| `ts` | T-stacking | Centroid dist < 5.0 Å, plane angle ≈ 90° ± 30°, psi ≈ 90° ± 45° |
| `pc` | Pi-cation | Centroid–cation dist < 6.0 Å, normal–cation angle < 60° |
| `vdw` | Van der Waals | Distance < r₁ + r₂ + 0.5 Å |
| `hp` | Hydrophobic | C/S atoms < 4.0 Å (ALA, PHE, GLY, ILE, LEU, PRO, VAL, TRP) |

Use `--itypes all` to compute all types. Default: `hb sb ps ts pc`.

All geometric criteria can be overridden:

```bash
ultracontacts contacts ... \
  --hbond-cutoff-dist 3.2 \
  --hbond-cutoff-ang 120 \
  --vdw-epsilon 0.3
```

## Output Formats

### Atomistic contacts Parquet

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int32 | Trajectory frame index |
| `itype` | string | Interaction type (`hb`, `sb`, `ps`, `ts`, `pc`, `vdw`, `hp`) |
| `atom1` | string | `chain:resname:resid:name` |
| `atom2` | string | `chain:resname:resid:name` |

### Frequency Parquet (long format)

| Column | Type | Description |
|--------|------|-------------|
| `itype` | string | Interaction type |
| `res1` | string | `chain:resname:resid` (lexicographically ≤ res2) |
| `res2` | string | `chain:resname:resid` |
| `frequency` | float64 | Fraction of frames where contact occurs |
| `count` | int64 | Number of frames |

Canonical residue ordering matches [getcontacts](https://github.com/getcontacts/getcontacts): `res1 ≤ res2` lexicographically.

### Condensed Parquet (wide format)

One row. Columns are residue pair names `"res1-res2"`. Values are the fraction of frames where **any** contact of **any** type exists between that pair. Suitable for direct use in ML/statistics pipelines:

```python
import polars as pl
df = pl.read_parquet("contacts_condensed.parquet")
# shape: (1, N_pairs)
# columns: ["A:ALA:1-A:ARG:5", "A:ALA:1-A:GLY:8", ...]
```

## Comparison with getcontacts

ultracontacts uses the same geometric criteria and residue-pair ordering as [getcontacts](https://github.com/getcontacts/getcontacts) and produces compatible output, with these differences:

- **VdW/HP sidechain contacts between adjacent residues**: ultracontacts reports these; getcontacts blanket-excludes all adjacent-residue VdW/HP pairs
- **Speed**: ~150 fps on GPU vs ~1–5 fps for getcontacts on CPU
- **No VMD dependency**: uses MDAnalysis + CuPy CUDA kernels

## Architecture

Static topology data (atom indices, masks, radii) is uploaded to GPU **once** at startup. Per-frame, only coordinate data (~70 KB) is transferred. Five fused CUDA raw kernels evaluate candidate pairs via `atomicAdd` without materialising the full N² distance matrix:

- **`dist_contacts`** — distance threshold (sb, hp)
- **`vdw_contacts`** — per-pair VdW cutoff from atomic radii (vdw)
- **`hbond_contacts`** — D···A distance + D-H···A angle at H vertex (hb)
- **`ring_stacking`** — centroid distance + plane angle + psi (ps, ts)
- **`pi_cation`** — centroid distance + normal-cation angle (pc)

Frequency calculations use **Polars lazy streaming** — bounded memory, Rust/SIMD string operations.
