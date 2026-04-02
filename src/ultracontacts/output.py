"""
output.py — Write contacts to Parquet and compute contact frequencies.

Parquet schema:
    frame  : int32
    itype  : string
    atom1  : string   "chain:resname:resid:atomname"
    atom2  : string   "chain:resname:resid:atomname"

Frequency TSV output (getcontacts-compatible):
    itype  res1  res2  frequency
where res1/res2 = "chain:resname:resid"  and  res1 <= res2  lexicographically.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema([
    pa.field("frame", pa.int32()),
    pa.field("itype", pa.string()),
    pa.field("atom1", pa.string()),
    pa.field("atom2", pa.string()),
])


def write_parquet_chunk(
    contacts: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    output_path: str,
    writer,
    itypes: list[str],
    beg: int,
    end: int,
    stride: int,
) -> pq.ParquetWriter:
    """Append a batch of contacts to the Parquet file."""
    frames, it_arr, a1_arr, a2_arr = contacts
    if len(frames) == 0:
        return writer

    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(frames, type=pa.int32()),
            pa.array(it_arr, type=pa.string()),
            pa.array(a1_arr, type=pa.string()),
            pa.array(a2_arr, type=pa.string()),
        ],
        schema=_SCHEMA,
    )

    if writer is None:
        writer = pq.ParquetWriter(output_path, _SCHEMA, compression="snappy")

    writer.write_batch(batch)
    return writer


def finalize_parquet(writer, output_path: str):
    """Close the ParquetWriter. Creates an empty file if no contacts were written."""
    if writer is not None:
        writer.close()
        return
    empty = pa.table(
        {"frame": pa.array([], type=pa.int32()),
         "itype": pa.array([], type=pa.string()),
         "atom1": pa.array([], type=pa.string()),
         "atom2": pa.array([], type=pa.string())},
    )
    pq.write_table(empty, output_path, compression="snappy")


# ---------------------------------------------------------------------------
# Contact frequency computation — Polars streaming
# ---------------------------------------------------------------------------

# Interaction types subject to adjacent-residue filtering
_ADJACENT_FILTER_ITYPES = {"vdw", "hp", "hplp", "hpll", "hppl"}


def _filter_adjacent_vdw_hp(q):
    """
    Remove VDW/HP contacts between adjacent residues on the same chain.

    Matches getcontacts behaviour (VDW_RES_DIFF=2 blanket exclusion for VDW/HP).
    Operates on a lazy frame that already has atom1/atom2 columns.
    """
    import polars as pl

    # Extract chain and resid from atom labels  ("A:ALA:124:CA" → chain="A", resid=124)
    q = q.with_columns([
        pl.col("atom1").str.split(":").list.get(0).alias("_chain1"),
        pl.col("atom1").str.split(":").list.get(2).cast(pl.Int32).alias("_resid1"),
        pl.col("atom2").str.split(":").list.get(0).alias("_chain2"),
        pl.col("atom2").str.split(":").list.get(2).cast(pl.Int32).alias("_resid2"),
    ])

    is_adjacent_vdw_hp = (
        pl.col("itype").is_in(list(_ADJACENT_FILTER_ITYPES))
        & (pl.col("_chain1") == pl.col("_chain2"))
        & ((pl.col("_resid1") - pl.col("_resid2")).abs() < 2)
    )

    q = q.filter(~is_adjacent_vdw_hp).drop(["_chain1", "_resid1", "_chain2", "_resid2"])
    return q


def compute_frequencies(
    parquet_path: str,
    output_path: Optional[str] = None,
    itype_filter: Optional[list[str]] = None,
    include_all: bool = False,
) -> list[tuple[str, str, str, float, int]]:
    """
    Compute residue-level contact frequencies from a contacts Parquet file.

    Uses Polars lazy streaming — processes the parquet in bounded memory with
    SIMD-vectorised Rust string ops. Matches getcontacts canonical ordering:
      resi1 = ":".join(atom1.split(":")[0:3])
      resi2 = ":".join(atom2.split(":")[0:3])
      if resi2 < resi1: resi1, resi2 = resi2, resi1  (standard lexicographic)

    Per-frame residue deduplication: multiple atom-level contacts between the
    same residue pair in one frame count as a single contact.
    total_frames = number of distinct frame values present in the file.

    output_path : str | None
        If given, write frequencies to this path.
        Extension determines format: .tsv → TSV text; anything else → Parquet.
    include_all : bool
        If False (default), adjacent-residue VDW/HP contacts are excluded
        (matching getcontacts VDW_RES_DIFF=2). If True, all contacts are kept.
    """
    import polars as pl

    q = pl.scan_parquet(parquet_path)

    if itype_filter:
        q = q.filter(pl.col("itype").is_in(itype_filter))

    # Filter adjacent-residue VDW/HP unless include_all
    if not include_all:
        q = _filter_adjacent_vdw_hp(q)

    # Count distinct frames (correct for strided/subsetted trajectories)
    total_frames = (
        q.select(pl.col("frame").n_unique())
        .collect()
        .item()
    )

    if total_frames == 0:
        if output_path:
            _write_frequencies([], output_path, 0, itype_filter)
        return []

    # Strip atom name → residue label "chain:resname:resid" (first 3 fields)
    # Mirrors: ":".join(atom.split(":")[0:3])
    q = q.with_columns([
        pl.col("atom1").str.split(":").list.slice(0, 3).list.join(":").alias("res1_raw"),
        pl.col("atom2").str.split(":").list.slice(0, 3).list.join(":").alias("res2_raw"),
    ])

    # Canonical ordering: res1 <= res2  (lexicographic — matches getcontacts)
    q = q.with_columns([
        pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
          .then(pl.col("res2_raw"))
          .otherwise(pl.col("res1_raw"))
          .alias("res1"),
        pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
          .then(pl.col("res1_raw"))
          .otherwise(pl.col("res2_raw"))
          .alias("res2"),
    ])

    # Deduplicate: one contact per (frame, itype, res1, res2)
    q = q.unique(subset=["frame", "itype", "res1", "res2"])

    # Count frames per (itype, res1, res2)
    result = (
        q.group_by(["itype", "res1", "res2"])
         .agg(pl.len().alias("count"))
         .with_columns((pl.col("count") / total_frames).alias("frequency"))
         .sort(["itype", "frequency"], descending=[False, True])
         .collect(streaming=True)
    )

    rows = [
        (r["itype"], r["res1"], r["res2"], r["frequency"], r["count"])
        for r in result.iter_rows(named=True)
    ]

    if output_path:
        _write_frequencies(rows, output_path, total_frames, itype_filter)

    return rows


def _write_frequencies(
    rows: list[tuple],
    output_path: str,
    total_frames: int,
    itype_filter: Optional[list[str]],
):
    """Write frequencies to parquet or TSV based on file extension."""
    if output_path.lower().endswith(".tsv"):
        _write_frequency_tsv(rows, output_path, total_frames, itype_filter)
    else:
        _write_frequency_parquet(rows, output_path, total_frames, itype_filter)


def _write_frequency_parquet(
    rows: list[tuple],
    output_path: str,
    total_frames: int,
    itype_filter: Optional[list[str]],
):
    import polars as pl
    if rows:
        df = pl.DataFrame({
            "itype":     [r[0] for r in rows],
            "res1":      [r[1] for r in rows],
            "res2":      [r[2] for r in rows],
            "frequency": [r[3] for r in rows],
            "count":     [r[4] for r in rows],
        })
    else:
        df = pl.DataFrame({
            "itype": pl.Series([], dtype=pl.Utf8),
            "res1":  pl.Series([], dtype=pl.Utf8),
            "res2":  pl.Series([], dtype=pl.Utf8),
            "frequency": pl.Series([], dtype=pl.Float64),
            "count":     pl.Series([], dtype=pl.Int64),
        })
    df.write_parquet(output_path, compression="snappy")
    print(f"[ultracontacts] Frequencies written to {output_path}  ({len(rows):,} pairs)")


def _write_frequency_tsv(
    rows: list[tuple],
    output_tsv: str,
    total_frames: int,
    itype_filter: Optional[list[str]],
):
    itype_str = ",".join(itype_filter) if itype_filter else "all"
    with open(output_tsv, "w") as fh:
        fh.write(f"#\ttotal_frames:{total_frames}\tinteraction_types:{itype_str}\n")
        fh.write("#\tColumns:\titype\tres1\tres2\tfrequency\n")
        for itype, res1, res2, freq, _count in rows:
            fh.write(f"{itype}\t{res1}\t{res2}\t{freq:.4f}\n")
    print(f"[ultracontacts] Frequencies written to {output_tsv}  ({len(rows):,} pairs)")


def default_freq_path(contacts_path: str) -> str:
    """Derive the default frequency output path (parquet) from the contacts path."""
    base, _ = os.path.splitext(contacts_path)
    return base + "_frequencies.parquet"


def default_condensed_path(contacts_path: str) -> str:
    """Derive the default condensed output path (parquet) from the contacts path."""
    base, _ = os.path.splitext(contacts_path)
    return base + "_condensed.parquet"


# ---------------------------------------------------------------------------
# Condensed frequency computation
# ---------------------------------------------------------------------------

def compute_condensed(
    parquet_path: str,
    output_path: Optional[str] = None,
    itype_filter: Optional[list[str]] = None,
    include_all: bool = False,
) -> dict[str, float]:
    """
    Compute a condensed (wide-format, single-row) contact probability table.

    Each column is a residue pair formatted as "res1-res2" where res1 ≤ res2
    lexicographically (matching getcontacts canonical ordering).  The single
    row value is the fraction of frames in which *any* contact of *any* type
    exists between those two residues.

    Output file:
      .parquet (default) — Polars single-row wide DataFrame
      .tsv               — tab-separated, header row + data row

    Returns dict mapping pair name → frequency.
    """
    import polars as pl

    q = pl.scan_parquet(parquet_path)

    if itype_filter:
        q = q.filter(pl.col("itype").is_in(itype_filter))

    # Filter adjacent-residue VDW/HP unless include_all
    if not include_all:
        q = _filter_adjacent_vdw_hp(q)

    total_frames = q.select(pl.col("frame").n_unique()).collect().item()

    if total_frames == 0:
        if output_path:
            _write_condensed({}, output_path)
        return {}

    # Residue labels — same canonical extraction as compute_frequencies
    q = q.with_columns([
        pl.col("atom1").str.split(":").list.slice(0, 3).list.join(":").alias("res1_raw"),
        pl.col("atom2").str.split(":").list.slice(0, 3).list.join(":").alias("res2_raw"),
    ]).with_columns([
        pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
          .then(pl.col("res2_raw")).otherwise(pl.col("res1_raw")).alias("res1"),
        pl.when(pl.col("res2_raw") < pl.col("res1_raw"))
          .then(pl.col("res1_raw")).otherwise(pl.col("res2_raw")).alias("res2"),
    ])

    # Deduplicate across ALL itypes: any contact between res pair in a frame = 1
    result = (
        q.unique(subset=["frame", "res1", "res2"])
         .group_by(["res1", "res2"])
         .agg(pl.len().alias("count"))
         .with_columns([
             (pl.col("count") / total_frames).alias("frequency"),
             (pl.col("res1") + "-" + pl.col("res2")).alias("pair"),
         ])
         .select(["pair", "frequency"])
         .sort("pair")            # lexicographic column order for consistency
         .collect(streaming=True)
    )

    pairs = result["pair"].to_list()
    freqs = result["frequency"].to_list()
    freq_map = dict(zip(pairs, freqs))

    if output_path:
        _write_condensed(freq_map, output_path)

    return freq_map


def _write_condensed(freq_map: dict[str, float], output_path: str):
    import polars as pl

    if freq_map:
        # Columns sorted lexicographically (guaranteed — pairs were sorted above)
        df = pl.DataFrame({pair: [freq] for pair, freq in freq_map.items()})
    else:
        df = pl.DataFrame()

    if output_path.lower().endswith(".tsv"):
        df.write_csv(output_path, separator="\t")
    else:
        df.write_parquet(output_path, compression="snappy")

    print(f"[ultracontacts] Condensed frequencies written to {output_path}"
          f"  ({len(freq_map):,} pairs)")


# ---------------------------------------------------------------------------
# Legacy TSV reader (for cross-validation with getcontacts output)
# ---------------------------------------------------------------------------

def read_getcontacts_tsv(tsv_path: str) -> list[dict]:
    """Parse a getcontacts-format TSV into a list of dicts matching our schema."""
    rows = []
    with open(tsv_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            frame, itype, atom1, atom2 = int(parts[0]), parts[1], parts[2], parts[3]
            atom1 = ":".join(atom1.split(":")[:4])
            atom2 = ":".join(atom2.split(":")[:4])
            rows.append({"frame": frame, "itype": itype, "atom1": atom1, "atom2": atom2})
    return rows
