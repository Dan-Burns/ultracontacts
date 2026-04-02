"""
cli.py — Command-line interface for ultracontacts.

Subcommands
-----------
ultracontacts contacts  --topology … --trajectory … --output contacts.parquet [--frequencies [freq.tsv]]
ultracontacts frequencies --input contacts.parquet --output freq.tsv

Legacy flat invocation (backwards-compat):
ultracontacts --topology … --trajectory … --output contacts.parquet
"""

from __future__ import annotations
import argparse
import os
import sys

from .runner import compute_contacts, DEFAULT_GEOM
from .output import compute_frequencies, default_freq_path


# ---------------------------------------------------------------------------
# Geometric criteria argument group (shared between subcommands)
# ---------------------------------------------------------------------------

def _add_geom_args(p: argparse.ArgumentParser):
    geo = p.add_argument_group("Geometric criteria overrides")
    geo.add_argument("--sb-cutoff-dist",    type=float, default=None, metavar="FLOAT",
                     help=f"Salt bridge distance cutoff [default: {DEFAULT_GEOM['SB_CUTOFF_DIST']} Å]")
    geo.add_argument("--pc-cutoff-dist",    type=float, default=None, metavar="FLOAT",
                     help=f"Pi-cation distance cutoff [default: {DEFAULT_GEOM['PC_CUTOFF_DIST']} Å]")
    geo.add_argument("--pc-cutoff-ang",     type=float, default=None, metavar="FLOAT",
                     help=f"Pi-cation angle cutoff [default: {DEFAULT_GEOM['PC_CUTOFF_ANG']}°]")
    geo.add_argument("--ps-cutoff-dist",    type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking distance cutoff [default: {DEFAULT_GEOM['PS_CUTOFF_DIST']} Å]")
    geo.add_argument("--ps-cutoff-ang",     type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking angle cutoff [default: {DEFAULT_GEOM['PS_CUTOFF_ANG']}°]")
    geo.add_argument("--ps-psi-ang",        type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking psi angle [default: {DEFAULT_GEOM['PS_PSI_ANG']}°]")
    geo.add_argument("--ts-cutoff-dist",    type=float, default=None, metavar="FLOAT",
                     help=f"T-stacking distance cutoff [default: {DEFAULT_GEOM['TS_CUTOFF_DIST']} Å]")
    geo.add_argument("--ts-cutoff-ang",     type=float, default=None, metavar="FLOAT",
                     help=f"T-stacking angle cutoff [default: {DEFAULT_GEOM['TS_CUTOFF_ANG']}°]")
    geo.add_argument("--ts-psi-ang",        type=float, default=None, metavar="FLOAT",
                     help=f"T-stacking psi angle [default: {DEFAULT_GEOM['TS_PSI_ANG']}°]")
    geo.add_argument("--hbond-cutoff-dist", type=float, default=None, metavar="FLOAT",
                     help=f"H-bond D···A cutoff [default: {DEFAULT_GEOM['HBOND_CUTOFF_DIST']} Å]")
    geo.add_argument("--hbond-cutoff-ang",  type=float, default=None, metavar="FLOAT",
                     help=f"H-bond D-H···A minimum angle [default: {DEFAULT_GEOM['HBOND_CUTOFF_ANG']}°]")
    geo.add_argument("--hbond-res-diff",    type=int,   default=None, metavar="INT",
                     help=f"H-bond min residue separation [default: {DEFAULT_GEOM['HBOND_RES_DIFF']}]")
    geo.add_argument("--vdw-epsilon",       type=float, default=None, metavar="FLOAT",
                     help=f"VdW padding [default: {DEFAULT_GEOM['VDW_EPSILON']} Å]")
    geo.add_argument("--vdw-res-diff",      type=int,   default=None, metavar="INT",
                     help=f"VdW min residue separation [default: {DEFAULT_GEOM['VDW_RES_DIFF']}]")
    geo.add_argument("--hp-cutoff-dist",    type=float, default=None, metavar="FLOAT",
                     help=f"Hydrophobic distance cutoff [default: {DEFAULT_GEOM['HP_CUTOFF_DIST']} Å]")


def _build_geom_overrides(args) -> dict:
    mapping = {
        "sb_cutoff_dist":   "SB_CUTOFF_DIST",
        "pc_cutoff_dist":   "PC_CUTOFF_DIST",
        "pc_cutoff_ang":    "PC_CUTOFF_ANG",
        "ps_cutoff_dist":   "PS_CUTOFF_DIST",
        "ps_cutoff_ang":    "PS_CUTOFF_ANG",
        "ps_psi_ang":       "PS_PSI_ANG",
        "ts_cutoff_dist":   "TS_CUTOFF_DIST",
        "ts_cutoff_ang":    "TS_CUTOFF_ANG",
        "ts_psi_ang":       "TS_PSI_ANG",
        "hbond_cutoff_dist": "HBOND_CUTOFF_DIST",
        "hbond_cutoff_ang":  "HBOND_CUTOFF_ANG",
        "hbond_res_diff":    "HBOND_RES_DIFF",
        "vdw_epsilon":       "VDW_EPSILON",
        "vdw_res_diff":      "VDW_RES_DIFF",
        "hp_cutoff_dist":    "HP_CUTOFF_DIST",
    }
    return {v: getattr(args, k) for k, v in mapping.items() if getattr(args, k, None) is not None}


# ---------------------------------------------------------------------------
# Subcommand: contacts
# ---------------------------------------------------------------------------

def _build_contacts_parser(p: argparse.ArgumentParser):
    p.add_argument("--topology",     metavar="PATH", required=True,
                   help="Path to topology file (.pdb, .psf, .prmtop, …)")
    p.add_argument("--trajectory",   metavar="PATH", default=None,
                   help="Path to trajectory file (.dcd, .xtc, .nc, …)")
    p.add_argument("--output",       metavar="PATH", required=True,
                   help="Output contacts file (.parquet)")
    p.add_argument("--itypes",       nargs="+", default=["hb", "sb", "ps", "ts", "pc"],
                   metavar="TYPE",
                   help="Interaction types: hb sb ps ts pc vdw hp all [default: hb sb ps ts pc]")
    p.add_argument("--sele",         metavar="STR", default="protein",
                   help="MDAnalysis selection 1 [default: 'protein']")
    p.add_argument("--sele2",        metavar="STR", default=None,
                   help="MDAnalysis selection 2 [default: same as --sele]")
    p.add_argument("--openmm-system", metavar="PATH", default=None,
                   help="OpenMM system.xml for exact bond topology")
    p.add_argument("--beg",          type=int, default=0, metavar="INT",
                   help="First frame [default: 0]")
    p.add_argument("--end",          type=int, default=None, metavar="INT",
                   help="Last frame [default: end of trajectory]")
    p.add_argument("--stride",       type=int, default=1, metavar="INT",
                   help="Frame stride [default: 1]")
    p.add_argument("--n-gpus",       type=int, default=None, metavar="INT",
                   help="Number of GPUs [default: 1]")
    p.add_argument(
        "--frequencies",
        nargs="?",          # 0 or 1 argument
        const="",           # --frequencies with no path → use auto path
        default=None,       # flag not provided → no frequency output
        metavar="TSV",
        help="Also compute contact frequencies after the contact calculation.\n"
             "Optionally provide the output TSV path; if omitted the file is\n"
             "written next to --output as <name>_frequencies.tsv",
    )
    _add_geom_args(p)


def _run_contacts(args):
    geom_overrides = _build_geom_overrides(args)

    compute_contacts(
        topology=args.topology,
        trajectory=args.trajectory,
        itypes=args.itypes,
        geom_criteria=geom_overrides or None,
        sele1=args.sele,
        sele2=args.sele2,
        beg=args.beg,
        end=args.end,
        stride=args.stride,
        output=args.output,
        openmm_system=args.openmm_system,
        n_gpus=args.n_gpus,
    )

    if args.frequencies is not None:
        freq_path = args.frequencies if args.frequencies else default_freq_path(args.output)
        print(f"[ultracontacts] Computing contact frequencies → {freq_path}")
        compute_frequencies(args.output, output_tsv=freq_path)


# ---------------------------------------------------------------------------
# Subcommand: frequencies
# ---------------------------------------------------------------------------

def _build_frequencies_parser(p: argparse.ArgumentParser):
    p.add_argument("--input",  metavar="PARQUET", required=True,
                   help="Path to contacts .parquet file")
    p.add_argument("--output", metavar="TSV", default=None,
                   help="Output TSV path [default: <input>_frequencies.tsv]")
    p.add_argument("--itype-filter", nargs="+", default=None, metavar="TYPE",
                   help="Only include these interaction types")


def _run_frequencies(args):
    out = args.output if args.output else default_freq_path(args.input)
    print(f"[ultracontacts] Computing contact frequencies → {out}")
    compute_frequencies(args.input, output_tsv=out, itype_filter=args.itype_filter)


# ---------------------------------------------------------------------------
# Root parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ultracontacts",
        description="GPU-accelerated molecular contact analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = root.add_subparsers(dest="command")

    # --- contacts subcommand ---
    p_contacts = sub.add_parser(
        "contacts",
        help="Compute per-frame atomic contacts from a trajectory",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _build_contacts_parser(p_contacts)

    # --- frequencies subcommand ---
    p_freq = sub.add_parser(
        "frequencies",
        help="Compute residue-level contact frequencies from a contacts .parquet file",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    _build_frequencies_parser(p_freq)

    return root


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "contacts":
        _run_contacts(args)
    elif args.command == "frequencies":
        _run_frequencies(args)
    else:
        # No subcommand — show help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
