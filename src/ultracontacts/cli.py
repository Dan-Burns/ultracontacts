"""
cli.py — Command-line interface for ultracontacts.

Usage:
    ultracontacts --topology protein.pdb --trajectory sim.dcd \
                  --itypes hb sb ps ts pc vdw hp \
                  --sele "protein" --sele2 "resname LIG" \
                  --output contacts.parquet \
                  --stride 1 --n-gpus 1

    # Compute frequencies from an existing contacts file:
    ultracontacts --frequencies contacts.parquet --output contacts_freq.tsv
"""

from __future__ import annotations
import argparse
import sys

from .runner import compute_contacts, DEFAULT_GEOM
from .output import compute_frequencies


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ultracontacts",
        description="GPU-accelerated chemical-group-aware molecular contact analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--topology", metavar="PATH",
                      help="Path to topology file (.pdb, .psf, .prmtop, …)")
    mode.add_argument("--frequencies", metavar="PARQUET",
                      help="Compute contact frequencies from an existing contacts.parquet file")

    p.add_argument("--trajectory", metavar="PATH", default=None,
                   help="Path to trajectory file (.dcd, .xtc, .nc, …) [optional for static PDB]")
    p.add_argument("--output", metavar="PATH", required=True,
                   help="Output file path (.parquet for contacts, .tsv for frequencies)")
    p.add_argument("--itypes", nargs="+", default=["hb", "sb", "ps", "ts", "pc"],
                   metavar="TYPE",
                   help="Interaction types to compute (default: hb sb ps ts pc)\n"
                        "  hb  = hydrogen bonds\n"
                        "  sb  = salt bridges\n"
                        "  ps  = pi-stacking\n"
                        "  ts  = T-stacking\n"
                        "  pc  = pi-cation\n"
                        "  vdw = van der Waals\n"
                        "  hp  = hydrophobic\n"
                        "  all = all of the above")
    p.add_argument("--sele", metavar="STR", default="protein",
                   help="MDAnalysis selection 1 [default: 'protein']")
    p.add_argument("--sele2", metavar="STR", default=None,
                   help="MDAnalysis selection 2 [default: same as --sele]")
    p.add_argument("--openmm-system", metavar="PATH", default=None,
                   help="Path to OpenMM system.xml to import exact bond topology")
    p.add_argument("--beg", type=int, default=0, metavar="INT",
                   help="First frame [default: 0]")
    p.add_argument("--end", type=int, default=None, metavar="INT",
                   help="Last frame [default: end of trajectory]")
    p.add_argument("--stride", type=int, default=1, metavar="INT",
                   help="Frame stride [default: 1]")
    p.add_argument("--n-gpus", type=int, default=None, metavar="INT",
                   help="Number of GPUs to use [default: all available]")
    p.add_argument("--chunk-size", type=int, default=500, metavar="INT",
                   help="Frames per GPU dispatch [default: 500]")

    # Geometric criteria overrides
    geo = p.add_argument_group("Geometric criteria overrides")
    geo.add_argument("--sb-cutoff-dist",  type=float, default=None, metavar="FLOAT",
                     help=f"Salt bridge distance cutoff [default: {DEFAULT_GEOM['SB_CUTOFF_DIST']} Å]")
    geo.add_argument("--pc-cutoff-dist",  type=float, default=None, metavar="FLOAT",
                     help=f"Pi-cation distance cutoff [default: {DEFAULT_GEOM['PC_CUTOFF_DIST']} Å]")
    geo.add_argument("--pc-cutoff-ang",   type=float, default=None, metavar="FLOAT",
                     help=f"Pi-cation angle cutoff [default: {DEFAULT_GEOM['PC_CUTOFF_ANG']}°]")
    geo.add_argument("--ps-cutoff-dist",  type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking distance cutoff [default: {DEFAULT_GEOM['PS_CUTOFF_DIST']} Å]")
    geo.add_argument("--ps-cutoff-ang",   type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking angle cutoff [default: {DEFAULT_GEOM['PS_CUTOFF_ANG']}°]")
    geo.add_argument("--ps-psi-ang",      type=float, default=None, metavar="FLOAT",
                     help=f"Pi-stacking psi angle [default: {DEFAULT_GEOM['PS_PSI_ANG']}°]")
    geo.add_argument("--ts-cutoff-dist",  type=float, default=None, metavar="FLOAT",
                     help=f"T-stacking distance cutoff [default: {DEFAULT_GEOM['TS_CUTOFF_DIST']} Å]")
    geo.add_argument("--ts-cutoff-ang",   type=float, default=None, metavar="FLOAT",
                     help=f"T-stacking angle cutoff [default: {DEFAULT_GEOM['TS_CUTOFF_ANG']}°]")
    geo.add_argument("--ts-psi-ang",      type=float, default=None, metavar="FLOAT",
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

    # Frequency mode options
    freq = p.add_argument_group("Frequency mode options (used with --frequencies)")
    freq.add_argument("--itype-filter", nargs="+", default=None, metavar="TYPE",
                      help="Filter to these interaction types when computing frequencies")

    return p


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
    overrides = {}
    for attr, key in mapping.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    return overrides


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.frequencies:
        # Frequency mode
        compute_frequencies(
            parquet_path=args.frequencies,
            output_tsv=args.output,
            itype_filter=args.itype_filter,
        )
        return

    # Contact computation mode
    geom_overrides = _build_geom_overrides(args)

    compute_contacts(
        topology=args.topology,
        trajectory=args.trajectory,
        itypes=args.itypes,
        geom_criteria=geom_overrides if geom_overrides else None,
        sele1=args.sele,
        sele2=args.sele2,
        beg=args.beg,
        end=args.end,
        stride=args.stride,
        output=args.output,
        openmm_system=args.openmm_system,
        n_gpus=args.n_gpus,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
