"""
output.py — Write contacts to Parquet and compute contact frequencies.

Parquet schema:
    frame  : int32
    itype  : string
    atom1  : string   "chain:resname:resid:name"
    atom2  : string   "chain:resname:resid:name"

Frequency TSV output:
    itype  res1  res2  frequency
where res1/res2 = "chain:resname:resid"
"""

from __future__ import annotations
import os
from collections import defaultdict
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
    writer,          # pq.ParquetWriter | None
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
# Contact frequency computation
# ---------------------------------------------------------------------------

def _atom_to_res(atom: str) -> str:
    """'A:ALA:1:CA'  →  'A:ALA:1'"""
    idx = atom.rfind(":")
    return atom[:idx] if idx != -1 else atom


def compute_frequencies(
    parquet_path: str,
    output_tsv: Optional[str] = None,
    itype_filter: Optional[list[str]] = None,
) -> list[tuple[str, str, str, float, int]]:
    """
    Compute residue-level contact frequencies from a contacts Parquet file.

    Uses a streaming row-group scan — memory stays flat regardless of file size.
    Per-frame residue deduplication mirrors getcontacts' res_contacts_xl logic:
    multiple atom-level contacts between the same residue pair within one frame
    count as a single contact.

    Parameters
    ----------
    parquet_path : str
        Path to the contacts .parquet file produced by ultracontacts.
    output_tsv : str | None
        If given, write tab-separated frequencies to this path.
    itype_filter : list[str] | None
        If given, only include these interaction types.

    Returns
    -------
    List of (itype, res1, res2, frequency, count) sorted by itype, frequency desc.
    """
    itype_set = set(itype_filter) if itype_filter else None

    pf = pq.ParquetFile(parquet_path)
    total_frames = 0

    # counts[(itype, res1, res2)] = number of frames in which this pair contacts
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for batch in pf.iter_batches(columns=["frame", "itype", "atom1", "atom2"]):
        frames_col = batch.column("frame").to_pylist()
        itype_col  = batch.column("itype").to_pylist()
        atom1_col  = batch.column("atom1").to_pylist()
        atom2_col  = batch.column("atom2").to_pylist()

        # Group rows by frame within this batch
        # Use a per-frame set to deduplicate at residue level
        frame_pairs: dict[int, dict[str, set[tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))

        for frame, itype, atom1, atom2 in zip(frames_col, itype_col, atom1_col, atom2_col):
            if itype_set and itype not in itype_set:
                continue
            res1 = _atom_to_res(atom1)
            res2 = _atom_to_res(atom2)
            # canonical order
            if res2 < res1:
                res1, res2 = res2, res1
            frame_pairs[frame][itype].add((res1, res2))

        # Accumulate into counts; track max frame for total_frames
        for frame, itype_dict in frame_pairs.items():
            if frame + 1 > total_frames:
                total_frames = frame + 1
            for itype, pairs in itype_dict.items():
                for res1, res2 in pairs:
                    counts[(itype, res1, res2)] += 1

    if total_frames == 0 or not counts:
        rows = []
    else:
        rows = [
            (itype, res1, res2, count / total_frames, count)
            for (itype, res1, res2), count in counts.items()
        ]
        rows.sort(key=lambda r: (r[0], -r[3]))  # itype asc, frequency desc

    if output_tsv:
        _write_frequency_tsv(rows, output_tsv, total_frames, itype_filter)

    return rows


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
    """Derive the default frequency output path from the contacts parquet path."""
    base, _ = os.path.splitext(contacts_path)
    return base + "_frequencies.tsv"


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
