"""
ultracontacts — GPU-accelerated, chemical-group-aware molecular contact analysis.

Supports: hydrogen bonds (hb), pi-stacking (ps), T-stacking (ts), pi-cation (pc),
          salt bridges (sb), van der Waals (vdw), hydrophobics (hp).

Quick start:
    from ultracontacts import compute_contacts
    compute_contacts("protein.pdb", "traj.dcd", itypes=["hb", "sb"], output="contacts.parquet")
"""

try:
    import cupy  # noqa: F401
except ImportError as e:
    raise ImportError(
        "ultracontacts requires a CUDA-version-specific CuPy wheel that pip cannot "
        "auto-select. Install the one matching your CUDA runtime, e.g.:\n\n"
        "  pip install cupy-cuda12x   # CUDA 12.x\n"
        "  pip install cupy-cuda13x   # CUDA 13.x\n\n"
        "Or use the package extra shorthand:\n\n"
        "  pip install ultracontacts[cuda12x]\n"
        "  pip install ultracontacts[cuda13x]\n"
    ) from e

from .runner import compute_contacts
from .output import compute_frequencies

__all__ = ["compute_contacts", "compute_frequencies"]
__version__ = "0.1.0"
