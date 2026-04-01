"""
output.py — Write contacts to Parquet (frame-by-frame) or TSV (frequencies).

Parquet schema:
    frame  : int32
    itype  : dictionary<int8, string>   (category — very compressed)
    atom1  : dictionary<int16, string>  (category — repeated labels deduplicated)
    atom2  : dictionary<int16, string>

Contact frequency TSV:
    itype  res1  res2  frequency
where res1/res2 = "chain:resname:resid"
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
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
    """
    Append a batch of contacts to the Parquet file.
    Creates the writer on first call (writer=None).
    """
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
    # Write empty file
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

def compute_frequencies(
    parquet_path: str,
    output_tsv: Optional[str] = None,
    itype_filter: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute residue-level contact frequencies from a contacts Parquet file.

    Returns a DataFrame with columns: itype, res1, res2, frequency, count.
    If output_tsv is given, also writes a tab-delimited text file.

    res1/res2 format: "chain:resname:resid"  (atom name stripped)
    """
    df = pd.read_parquet(parquet_path)

    if itype_filter:
        df = df[df["itype"].isin(itype_filter)]

    if df.empty:
        freq_df = pd.DataFrame(columns=["itype", "res1", "res2", "count", "frequency"])
    else:
        total_frames = df["frame"].nunique()

        # Strip atom name to get residue label
        df["res1"] = df["atom1"].str.rsplit(":", n=1).str[0]
        df["res2"] = df["atom2"].str.rsplit(":", n=1).str[0]

        # Count unique contacts per (frame, itype, res1, res2) to avoid
        # double-counting same residue pair via different atoms
        dedup = df.drop_duplicates(subset=["frame", "itype", "res1", "res2"])

        freq_df = (
            dedup.groupby(["itype", "res1", "res2"])
            .size()
            .reset_index(name="count")
        )
        freq_df["frequency"] = freq_df["count"] / total_frames
        freq_df = freq_df.sort_values(
            ["itype", "frequency"], ascending=[True, False]
        ).reset_index(drop=True)

    if output_tsv:
        freq_df.to_csv(output_tsv, sep="\t", index=False,
                       columns=["itype", "res1", "res2", "frequency"])
        print(f"[ultracontacts] Frequencies written to {output_tsv}")

    return freq_df


# ---------------------------------------------------------------------------
# Legacy TSV reader (for compatibility with getcontacts output)
# ---------------------------------------------------------------------------

def read_getcontacts_tsv(tsv_path: str) -> pd.DataFrame:
    """
    Parse a getcontacts-format TSV into a DataFrame matching our schema.
    Useful for cross-validating ultracontacts vs getcontacts output.
    """
    rows = []
    with open(tsv_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            frame, itype, atom1, atom2 = int(parts[0]), parts[1], parts[2], parts[3]
            # Strip VMD index (last colon-separated field)
            atom1 = ":".join(atom1.split(":")[:4])
            atom2 = ":".join(atom2.split(":")[:4])
            rows.append({"frame": frame, "itype": itype, "atom1": atom1, "atom2": atom2})
    return pd.DataFrame(rows)
