# ultracontacts — Agent Context

## What This Is

GPU-accelerated molecular contact analysis in Python.  Computes H-bonds,
salt bridges, pi-stacking, T-stacking, pi-cation, VdW, and hydrophobic
contacts across MD trajectories.  Outputs Apache Parquet files.  Requires
a CUDA GPU (uses CuPy raw kernels — **no JAX**).

---

## CLI Subcommands

```
ultracontacts contacts   \
    --topology <file>  --trajectory <file>  --output contacts.parquet
    [--sele STR]  [--sele2 STR]  [--itypes hb sb ps ts pc vdw hp]
    [--beg N]  [--end N]  [--stride N]
    [--no-frequencies]                    # skip auto frequency output
    [--frequencies PATH]                  # custom path for frequency file
    [--condensed [PATH]]                  # also write condensed wide-format
    [--include-all]                       # keep adjacent-residue VdW/HP in frequencies
    [--openmm-system system.xml]          # use OpenMM bond topology
    [geometric override flags...]

ultracontacts frequencies  \
    --input contacts.parquet  [--output PATH]
    [--condensed]             # output condensed format instead of long format
    [--include-all]           # keep adjacent-residue VdW/HP contacts
    [--itype-filter hb sb ...]

```

**Default outputs of `contacts`:**
- `contacts.parquet`                — atomistic contacts (frame, itype, atom1, atom2)
- `contacts_frequencies.parquet`   — per-(itype,res1,res2) frequency (auto, disable with `--no-frequencies`)

**`frequencies` output formats:**
- Without `--condensed`: long format `(itype, res1, res2, frequency, count)`
- With `--condensed`: **wide format**, one row, columns = `"res1-res2"` pair names, values = any-itype contact probability

---

## Key Architecture

### Kernels (`kernels.py`)
Five `cupy.RawKernel` CUDA kernels, one thread per candidate pair, sparse
output via `atomicAdd`.  The full N² matrix is never materialised.

- `dist_contacts`  — distance threshold (sb, hp)
- `vdw_contacts`   — per-pair VdW cutoff from atomic radii (vdw)
- `hbond_contacts` — D···A distance + D-H···A angle at H vertex (hb)
- `ring_stacking`  — centroid dist + plane angle + psi (ps, ts)
- `pi_cation`      — centroid dist + normal–cation angle (pc)

### Topology parsing (`topology.py`)
- `parse_topology(u, sele1, sele2)` → `ChemicalGroups` dataclass
- Static data (indices, masks, radii) uploaded to GPU **once** at startup
- Bonds come from **OpenMM `system.xml`** (`--openmm-system`); if absent,
  falls back to `u.atoms.guess_bonds()`.  PDB CONECT records are unreliable —
  always pass `--openmm-system` for H-bond detection.

### Residue diff masks (`build_residue_diff_mask`)
Controls which atom pairs are excluded due to proximity along the chain.

| Interaction | Behaviour |
|---|---|
| hb | `HBOND_RES_DIFF=1`: skip same-residue only; backbone-backbone\* |
| sb, ps, ts, pc | No residue diff filter |
| vdw | `adjacent_bb_only=True`: skip same-residue entirely; adjacent only if **both** backbone |
| hp | Same as vdw |

\*For hb, `bb_a`/`bb_b` provided but `adjacent_bb_only=False` — meaning only
backbone-backbone pairs are filtered at the hbond_res_diff distance.
Sidechain H-bonds between adjacent residues ARE computed.

This means ultracontacts **captures** VdW/HP sidechain contacts between adjacent
residues (getcontacts does not — it blanket-excludes with `VDW_RES_DIFF=2`).

**However**, these are **filtered out by default** at frequency time (in `output.py`)
via `_filter_adjacent_vdw_hp()`.  Pass `--include-all` to keep them.

This two-layer design means:
- The atomistic parquet always has the full contact set (including adjacent sidechain VdW/HP)
- The frequency output matches getcontacts by default
- `--include-all` gives access to the extended contact set

### Frequency calculation (`output.py`)
Uses **Polars lazy streaming** (`collect(streaming=True)`) — bounded memory,
SIMD Rust string ops.

**Adjacent-residue filter** (`_filter_adjacent_vdw_hp`):
By default, rows with `itype in {vdw, hp, hplp, hpll, hppl}` where
`chain1 == chain2 AND |resid1-resid2| < 2` are dropped.  The filter parses
chain/resid from atom labels inline in the Polars lazy plan.  Disabled by
`include_all=True`.

Canonical residue-pair ordering matches getcontacts exactly:
```
res1 = ":".join(atom1.split(":")[:3])   # "chain:resname:resid"
res2 = ":".join(atom2.split(":")[:3])
if res2 < res1: res1, res2 = res2, res1  # lexicographic swap
```

`total_frames = frame.n_unique()` — correct for strided trajectories.

---

## Default Geometric Criteria

These match getcontacts defaults unless overridden:

| Parameter | Value | Notes |
|---|---|---|
| `HBOND_CUTOFF_DIST` | 3.5 Å | D···A distance |
| `HBOND_CUTOFF_ANG` | 110° | D-H···A minimum angle (VMD uses 70° deviation from 180°, i.e. ≡ 110°) |
| `HBOND_RES_DIFF` | 1 | Skip same-residue pairs |
| `SB_CUTOFF_DIST` | 4.0 Å | |
| `VDW_EPSILON` | 0.5 Å | Padding on top of r₁+r₂ |
| `VDW_RES_DIFF` | 2 | Used in `adjacent_bb_only` mode |
| `HP_CUTOFF_DIST` | 4.0 Å | C/S–C/S distance |
| `HP_RES_DIFF` | 2 | Used in `adjacent_bb_only` mode |
| PS/TS/PC | see runner.py | Match getcontacts |

**Critical:** getcontacts uses VMD `measure hbonds` with `cutoff_angle=70`
which means deviation from linearity — equivalent to D-H-A ≥ 110°.
Our kernel receives the minimum D-H-A angle directly.

---

## Validation Status (as of 2026-04-01)

Tested on a 6354-frame gamma-secretase trajectory (~20K atoms):

| | Count |
|---|---|
| Shared contacts (identical frequency) | ~9,800 |
| In getcontacts only | 4 |
| In ultracontacts only | ~227 (adjacent-residue sidechain VdW/HP) |

**4 contacts getcontacts finds but ultracontacts misses:**
- 3× adjacent GLY-GLY (unclear — possibly backbone CA-CA VdW thresholding)
- 1× PHE:143–PHE:70 (non-adjacent, possibly pi-stacking edge case)

**227 contacts ultracontacts finds but getcontacts misses:**
- Adjacent-residue sidechain VdW/HP contacts (intentional — see above)
- Sub 10⁻³ frequency HP contacts between aliphatic residues (likely real,
  just below getcontacts' effective threshold from its VMD selection)
- Disulfide CYS-CYS pairs: **getcontacts skips them for VdW; we do not yet**
  → TODO: add disulfide exclusion to `precompute_vdw_mask` in topology.py

---

## Known Issues / TODOs

1. **Disulfide CYS exclusion from VDW**: getcontacts skips CYS-CYS VdW for
   disulfide-bonded pairs (done for HP, missing for VdW).  Add the same
   `disulfide_resids` check to `precompute_vdw_mask` in `vanderwaals.py`.

2. **GLY-GLY adjacent contacts**: 3 adjacent GLY-GLY H-bond contacts are
   found by getcontacts but not ultracontacts.  Likely a backbone H N···O
   interaction — worth investigating the H-bond mask logic for GLY specifically.

3. **PHE-PHE non-adjacent miss**: 1 distant PHE:143–PHE:70 contact found by
   getcontacts but not ultracontacts.  Likely a pi-stacking edge case at the
   angle/psi boundary — investigate `ring_stacking` kernel threshold logic.

4. **Bond detection fallback**: If no `--openmm-system` is provided, `runner.py`
   now checks whether the bond count is reasonable (>= N_atoms/2).  PDB files
   with only a few CONECT records are detected as incomplete and `guess_bonds()`
   is called.  If that also fails, the user is warned to pass `--openmm-system`.
   MDAnalysis `guess_bonds()` can be slow and inaccurate on large systems.

5. **CUDA 13 + cuDF**: RAPIDS cuDF does not yet support CUDA 13.x.  The
   frequency pipeline uses Polars (CPU, Rust/SIMD) instead.  Once cuDF
   supports CUDA 13, replacing the Polars path with cuDF would give GPU-native
   string ops for the frequency pipeline.

6. **Spatial hashing / cell lists**: Current O(N²) pair enumeration works well
   for systems up to ~20K atoms at 150fps.  For larger systems (>30K atoms),
   a CUDA cell-list implementation would reduce to O(N).

7. **ParmEd-based universal bond loader**: To support GROMACS (.top), AMBER
   (.prmtop), CHARMM (.psf), and other topology formats without needing
   `--openmm-system`, add a `--parmed-topology` flag or auto-detect format.
   ParmEd can parse all major MD formats and expose `.bonds` as atom index
   pairs.  This would replace the current OpenMM XML parser with a single
   unified path: `parmed.load_file(path).bonds → MDAnalysis bonds`.

---

## Dependencies

- `cupy` — CUDA kernels (must match CUDA version; NOT cuDF/RAPIDS)
- `MDAnalysis` — trajectory parsing
- `pyarrow` — Parquet I/O
- `polars` — frequency/condensed calculation (streaming, SIMD)
- `tqdm` — progress bar
- `numpy` — CPU array ops

**Environment:** `ultracontacts` conda environment
