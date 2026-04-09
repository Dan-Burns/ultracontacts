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
from .output import (
    compute_frequencies, default_freq_path,
    compute_condensed, default_condensed_path,
)


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
    p.add_argument("--bond-topology", metavar="PATH", default=None,
                   help="Topology file with bond info for H-bond detection.\n"
                        "Auto-detected formats:\n"
                        "  OpenMM  : system.xml\n"
                        "  AMBER   : .prmtop, .parm7\n"
                        "  GROMACS : .top, .tpr\n"
                        "  CHARMM  : .psf\n"
                        "Full-system files (with solvent) work correctly against\n"
                        "solvent-stripped structures.")
    p.add_argument("--openmm-system", metavar="PATH", default=None,
                   help="[DEPRECATED — use --bond-topology] OpenMM system.xml")
    p.add_argument("--beg",          type=int, default=0, metavar="INT",
                   help="First frame [default: 0]")
    p.add_argument("--end",          type=int, default=None, metavar="INT",
                   help="Last frame [default: end of trajectory]")
    p.add_argument("--stride",       type=int, default=1, metavar="INT",
                   help="Frame stride [default: 1]")
    p.add_argument("--n-gpus",       type=int, default=None, metavar="INT",
                   help="Number of GPUs [default: 1]")

    freq_grp = p.add_mutually_exclusive_group()
    freq_grp.add_argument(
        "--no-frequencies",
        action="store_true",
        help="Skip frequency calculation (frequencies are computed by default)",
    )
    freq_grp.add_argument(
        "--frequencies",
        metavar="PATH",
        default=None,
        help="Custom path for the frequency output file.\n"
             "  .parquet extension → Parquet (default if omitted)\n"
             "  .tsv extension     → tab-separated text\n"
             "  default path: <contacts>_frequencies.parquet",
    )
    p.add_argument(
        "--condensed",
        nargs="?",
        const="",          # --condensed with no path → auto path
        default=None,      # flag absent → no condensed output
        metavar="PATH",
        help="Also write a condensed wide-format frequency file.\n"
             "One row, columns = 'res1-res2' pair names, values = contact probability\n"
             "  (any itype, any frame where contact exists counts once).\n"
             "  .parquet extension → Parquet (default); .tsv → TSV\n"
             "  default path: <contacts>_condensed.parquet",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include adjacent-residue VdW/HP sidechain contacts in frequency output.\n"
             "By default these are excluded to match getcontacts behaviour.",
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
        bond_topology=args.bond_topology,
        n_gpus=args.n_gpus,
    )

    if not args.no_frequencies:
        freq_path = args.frequencies if args.frequencies else default_freq_path(args.output)
        print(f"[ultracontacts] Computing contact frequencies → {freq_path}")
        compute_frequencies(args.output, output_path=freq_path, include_all=args.include_all)

    if args.condensed is not None:
        cond_path = args.condensed if args.condensed else default_condensed_path(args.output)
        print(f"[ultracontacts] Computing condensed frequencies → {cond_path}")
        compute_condensed(args.output, output_path=cond_path, include_all=args.include_all)


# ---------------------------------------------------------------------------
# Subcommand: frequencies
# ---------------------------------------------------------------------------

def _build_frequencies_parser(p: argparse.ArgumentParser):
    p.add_argument("--input",  metavar="PARQUET", required=True,
                   help="Path to contacts .parquet file")
    p.add_argument("--output", metavar="PATH", default=None,
                   help="Output path.  Format determined by extension and --condensed flag.\n"
                        "  default (no --condensed): <input>_frequencies.parquet\n"
                        "  with --condensed:         <input>_condensed.parquet\n"
                        "  .parquet → Parquet (default); .tsv → tab-separated text")
    p.add_argument("--itype-filter", nargs="+", default=None, metavar="TYPE",
                   help="Only include these interaction types")
    p.add_argument(
        "--condensed",
        action="store_true",
        help="Output condensed wide-format: one row, columns = 'res1-res2' pair names,\n"
             "values = probability of any contact between that pair (itype-agnostic).\n"
             "Controls the format written to --output (does not add a second file).",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include adjacent-residue VdW/HP sidechain contacts.\n"
             "By default these are excluded to match getcontacts behaviour.",
    )


def _run_frequencies(args):
    if args.condensed:
        out = args.output if args.output else default_condensed_path(args.input)
        print(f"[ultracontacts] Computing condensed frequencies → {out}")
        compute_condensed(args.input, output_path=out, itype_filter=args.itype_filter,
                          include_all=args.include_all)
    else:
        out = args.output if args.output else default_freq_path(args.input)
        print(f"[ultracontacts] Computing contact frequencies → {out}")
        compute_frequencies(args.input, output_path=out, itype_filter=args.itype_filter,
                            include_all=args.include_all)


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
