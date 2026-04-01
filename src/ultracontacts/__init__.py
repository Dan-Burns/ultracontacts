"""
ultracontacts — GPU-accelerated, chemical-group-aware molecular contact analysis.

Supports: hydrogen bonds (hb), pi-stacking (ps), T-stacking (ts), pi-cation (pc),
          salt bridges (sb), van der Waals (vdw), hydrophobics (hp).

Quick start:
    from ultracontacts import compute_contacts
    compute_contacts("protein.pdb", "traj.dcd", itypes=["hb", "sb"], output="contacts.parquet")
"""

from .runner import compute_contacts
from .output import compute_frequencies

__all__ = ["compute_contacts", "compute_frequencies"]
__version__ = "0.1.0"
