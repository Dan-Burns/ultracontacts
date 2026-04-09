"""
topology.py — Parse MDAnalysis Universe into pre-computed chemical group index arrays.

All arrays are computed once from the topology and reused across every trajectory frame.
"""

from __future__ import annotations
import numpy as np
import MDAnalysis as mda
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Chemical group definitions — matching getcontacts exactly
# ---------------------------------------------------------------------------
AROMATIC_DEFS: dict[str, list[str]] = {
    "PHE": ["CG", "CE1", "CE2"],
    "TRP": ["CD2", "CZ2", "CZ3"],
    "TYR": ["CG", "CE1", "CE2"],
    "HIS": ["CG", "CE1", "CD2"],
    "HSD": ["CG", "CE1", "CD2"],
    "HSE": ["CG", "CE1", "CD2"],
    "HSP": ["CG", "CE1", "CD2"],
    "HIE": ["CG", "CE1", "CD2"],
    "HIP": ["CG", "CE1", "CD2"],
    "HID": ["CG", "CE1", "CD2"],
}

# Nucleic acid aromatic rings — one triad per ring, purines have 2 fused rings.
# Atom names follow standard AMBER/CHARMM nucleic naming.
# Each entry: (resname_variants, [triad1_names, triad2_names, ...])
NUCLEIC_RING_DEFS: list[tuple[frozenset, list[list[str]]]] = [
    # Adenine: 6-membered (C2,C4,C6) + 5-membered (C4,C5,N7)
    (frozenset({"ADE", "DA", "DA3", "DA5", "A"}),
     [["C2", "C4", "C6"], ["C4", "C5", "N7"]]),
    # Guanine: 6-membered (C2,C4,C6) + 5-membered (C4,C5,N7)
    (frozenset({"GUA", "DG", "DG3", "DG5", "G"}),
     [["C2", "C4", "C6"], ["C4", "C5", "N7"]]),
    # Cytosine: 6-membered only
    (frozenset({"CYT", "DC", "DC3", "DC5", "C"}),
     [["C2", "C4", "C6"]]),
    # Thymine: 6-membered only
    (frozenset({"THY", "DT", "DT3", "DT5", "T"}),
     [["C2", "C4", "C6"]]),
    # Uracil: 6-membered only
    (frozenset({"URA", "URI", "U"}),
     [["C2", "C4", "C6"]]),
]

ANION_DEFS: dict[str, list[str]] = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}
NUCLEIC_ANION_NAMES = {"OP1", "OP2", "O1P", "O2P"}

CATION_DEFS: dict[str, list[str]] = {
    "ARG": ["NH1", "NH2"],
    "LYS": ["NZ"],
    "HIS": ["ND1", "NE2"],
    "HSD": ["ND1", "NE2"],
    "HSE": ["ND1", "NE2"],
    "HSP": ["ND1", "NE2"],
    "HIE": ["ND1", "NE2"],
    "HIP": ["ND1", "NE2"],
    "HID": ["ND1", "NE2"],
}

HYDROPHOBIC_RESNAMES = frozenset(
    {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "TRP", "PHE", "CYS", "CYX"}
)

BACKBONE_NAMES = frozenset({"N", "CA", "C", "O", "H", "HA", "HN"})

# Standard VdW radii (Å) by element
VDW_RADII: dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
}
VDW_DEFAULT = 1.70


# ---------------------------------------------------------------------------
# Bond topology parsing — multi-engine support
# ---------------------------------------------------------------------------
def parse_openmm_bonds(xml_path: str, max_atoms: int | None = None) -> list[tuple[int, int]]:
    """
    Parse an OpenMM system.xml file to extract bond topology.
    Filters out any bonds referencing atom indices >= max_atoms, allowing
    a full system.xml to gracefully apply to a solvent-stripped PDB.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bonds = set()

    # Constraints (often handles X-H bonds)
    for c in root.findall(".//Constraint"):
        p1, p2 = c.get("p1"), c.get("p2")
        if p1 is not None and p2 is not None:
            i, j = int(p1), int(p2)
            if max_atoms is None or (i < max_atoms and j < max_atoms):
                bonds.add((min(i, j), max(i, j)))

    # Standard forces (HarmonicBondForce, CustomBondForce)
    for b in root.findall(".//Bond"):
        p1, p2 = b.get("p1"), b.get("p2")
        if p1 is not None and p2 is not None:
            i, j = int(p1), int(p2)
            if max_atoms is None or (i < max_atoms and j < max_atoms):
                bonds.add((min(i, j), max(i, j)))

    return list(bonds)


def parse_parmed_bonds(topo_path: str, max_atoms: int | None = None) -> list[tuple[int, int]]:
    """
    Parse bond topology from any ParmEd-supported file format.

    Supports AMBER (.prmtop, .parm7), GROMACS (.top, .tpr), CHARMM (.psf),
    Desmond (.cms), and others.

    Filters out bonds referencing atom indices >= max_atoms, so a full-system
    topology file works correctly against a solvent-stripped structure.

    Parameters
    ----------
    topo_path : str
        Path to the topology file.
    max_atoms : int, optional
        If given, discard bonds involving atom indices >= this value.

    Returns
    -------
    list[tuple[int, int]]
        Sorted (i, j) bond pairs with i < j.
    """
    try:
        import parmed
    except ImportError:
        raise ImportError(
            "ParmEd is required to read bond topology from this file format.\n"
            "Install it with:  pip install parmed\n"
            "  or:  conda install -c conda-forge parmed"
        )

    struct = parmed.load_file(topo_path)
    bonds = set()
    for bond in struct.bonds:
        i, j = bond.atom1.idx, bond.atom2.idx
        if max_atoms is None or (i < max_atoms and j < max_atoms):
            bonds.add((min(i, j), max(i, j)))
    return list(bonds)


# Extensions handled by the built-in OpenMM XML parser (no parmed needed)
_OPENMM_EXTENSIONS = frozenset({".xml"})


def parse_bond_topology(topo_path: str, max_atoms: int | None = None) -> list[tuple[int, int]]:
    """
    Unified bond-topology loader — auto-detects format from file extension.

    - ``.xml``  → OpenMM system.xml parser (stdlib only, no parmed)
    - Everything else → ParmEd (AMBER .prmtop, GROMACS .top/.tpr, CHARMM .psf, …)

    The ``max_atoms`` filter ensures that a full-system topology file
    (including solvent) works correctly against a solvent-stripped structure.
    """
    import os
    ext = os.path.splitext(topo_path)[1].lower()

    if ext in _OPENMM_EXTENSIONS:
        return parse_openmm_bonds(topo_path, max_atoms=max_atoms)
    else:
        return parse_parmed_bonds(topo_path, max_atoms=max_atoms)



# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class ChemicalGroups:
    """All pre-computed chemical group index arrays for a topology."""

    n_atoms: int  # total atoms in the Universe

    # Aromatic rings
    ring_indices: np.ndarray        # (R, 3)  int32 — 3 atoms per ring
    ring_labels: list[str]          # (R,) representative atom label per ring
    ring_in_sele1: np.ndarray       # (R,) bool
    ring_in_sele2: np.ndarray       # (R,) bool

    # H-bond donors (heavy atom + bonded H)
    donor_indices: np.ndarray       # (D,) int32 — heavy atom indices
    hydrogen_indices: np.ndarray    # (D,) int32 — H atom indices
    donor_labels: list[str]         # (D,)
    donor_in_sele1: np.ndarray      # (D,) bool
    donor_in_sele2: np.ndarray      # (D,) bool
    donor_is_bb: np.ndarray         # (D,) bool
    donor_chain: np.ndarray         # (D,) object array of chain IDs
    donor_resid: np.ndarray         # (D,) int32
    donor_is_ligand: np.ndarray     # (D,) bool

    # H-bond acceptors
    acceptor_indices: np.ndarray    # (A,) int32
    acceptor_labels: list[str]      # (A,)
    acceptor_in_sele1: np.ndarray   # (A,) bool
    acceptor_in_sele2: np.ndarray   # (A,) bool
    acceptor_is_bb: np.ndarray      # (A,) bool
    acceptor_chain: np.ndarray      # (A,) object
    acceptor_resid: np.ndarray      # (A,) int32
    acceptor_is_ligand: np.ndarray  # (A,) bool

    # Salt bridge anions / cations
    anion_indices: np.ndarray       # (Na,)
    anion_labels: list[str]
    anion_in_sele1: np.ndarray
    anion_in_sele2: np.ndarray
    anion_is_ligand: np.ndarray

    cation_indices: np.ndarray      # (Nc,)
    cation_labels: list[str]
    cation_in_sele1: np.ndarray
    cation_in_sele2: np.ndarray
    cation_is_ligand: np.ndarray

    # Hydrophobics
    hp_indices: np.ndarray          # (Hp,)
    hp_labels: list[str]
    hp_in_sele1: np.ndarray
    hp_in_sele2: np.ndarray
    hp_chain: np.ndarray
    hp_resid: np.ndarray
    hp_is_bb: np.ndarray            # (Hp,) bool
    hp_radii: np.ndarray            # (Hp,) float32 — VDW radii for per-pair cutoff

    # VdW — all heavy atoms in each selection
    vdw_indices1: np.ndarray        # (V1,)
    vdw_labels1: list[str]
    vdw_radii1: np.ndarray          # (V1,) float32
    vdw_chain1: np.ndarray
    vdw_resid1: np.ndarray
    vdw_is_ligand1: np.ndarray
    vdw_is_bb1: np.ndarray          # (V1,) bool

    vdw_indices2: np.ndarray        # (V2,)
    vdw_labels2: list[str]
    vdw_radii2: np.ndarray          # (V2,) float32
    vdw_chain2: np.ndarray
    vdw_resid2: np.ndarray
    vdw_is_ligand2: np.ndarray
    vdw_is_bb2: np.ndarray          # (V2,) bool

    # Per-atom info (indexed by Universe atom index)
    is_backbone: np.ndarray         # (N,) bool
    is_ligand: np.ndarray           # (N,) bool
    all_labels: list[str]           # (N,) "chain:resname:resid:name"

    disulfide_resids: set[tuple]    # {(chain, resid), ...} of SS-bonded CYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _atom_label(atom) -> str:
    return f"{atom.chainID}:{atom.resname}:{atom.resid}:{atom.name}"


def _to_i32(lst) -> np.ndarray:
    return np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32)


def _to_bool(lst) -> np.ndarray:
    return np.array(lst, dtype=bool) if lst else np.empty(0, dtype=bool)


def _get_vdw(element: str) -> float:
    return VDW_RADII.get(element.upper(), VDW_DEFAULT)


def _collect_rdkit_rings(
    lig_ag,
    sele1_set: set,
    sele2_set: set,
    ring_idx_rows: list,
    ring_lbls: list,
    ring_m1: list,
    ring_m2: list,
) -> None:
    """
    Detect aromatic rings in a ligand AtomGroup using RDKit's SSSR algorithm.

    For each unique 5- or 6-membered aromatic ring, selects 3 maximally-spaced
    atoms as the representative triad for centroid + normal computation, then
    appends to the shared ring lists.

    Silently returns if RDKit is not installed or the ligand has no bonds.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return

    # Process each residue separately so multi-residue ligands work
    for res in lig_ag.residues:
        atoms = list(res.atoms)
        if len(atoms) < 3:
            continue

        # Build local index map: RDKit atom idx → universe atom idx
        local_to_global = {i: a.index for i, a in enumerate(atoms)}
        name_map = {a.index: a for a in atoms}

        # Build RDKit mol from scratch (edit mol — no SMILES needed)
        em = Chem.RWMol()
        for atom in atoms:
            elem = atom.element.capitalize() if atom.element else "C"
            try:
                rd_atom = Chem.Atom(elem)
            except Exception:
                rd_atom = Chem.Atom("C")
            em.AddAtom(rd_atom)

        # Add bonds from MDAnalysis topology
        global_to_local = {a.index: i for i, a in enumerate(atoms)}
        added = set()
        try:
            for bond in res.atoms.bonds:
                a1_idx = bond.atoms[0].index
                a2_idx = bond.atoms[1].index
                local1 = global_to_local.get(a1_idx)
                local2 = global_to_local.get(a2_idx)
                if local1 is None or local2 is None:
                    continue
                key = (min(local1, local2), max(local1, local2))
                if key not in added:
                    em.AddBond(local1, local2, Chem.BondType.SINGLE)
                    added.add(key)
        except Exception:
            return  # No bond info

        try:
            mol = em.GetMol()
            Chem.SanitizeMol(mol)
        except Exception:
            return

        # Get smallest set of smallest rings
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        seen_rings = set()
        for ring in atom_rings:
            n = len(ring)
            if n not in (5, 6):
                continue

            # Check aromaticity: majority of ring atoms should be aromatic
            aromatic_count = sum(1 for ri in ring if mol.GetAtomWithIdx(ri).GetIsAromatic())
            if aromatic_count < n - 1:
                continue

            # Canonical ring key to deduplicate fused rings
            ring_key = frozenset(ring)
            if ring_key in seen_rings:
                continue
            seen_rings.add(ring_key)

            # Pick 3 maximally-spaced atoms: first, middle, and ~2/3 around ring
            ring_list = list(ring)
            if n == 6:
                triad_local = [ring_list[0], ring_list[2], ring_list[4]]
            else:  # 5-membered
                triad_local = [ring_list[0], ring_list[1], ring_list[3]]

            global_idxs = [local_to_global[i] for i in triad_local]

            ring_idx_rows.append(global_idxs)
            ring_lbls.append(_atom_label(name_map[global_idxs[0]]))
            ring_m1.append(any(gi in sele1_set for gi in global_idxs))
            ring_m2.append(any(gi in sele2_set for gi in global_idxs))



# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def parse_topology(
    universe: mda.Universe,
    sele1_str: str,
    sele2_str: str,
) -> ChemicalGroups:
    """
    Build ChemicalGroups from an MDAnalysis Universe.

    Bonds must be present in the topology (PSF, PRMTOP, MOL2, or guessed).
    For PDB files call u.atoms.guess_bonds() before passing.
    """
    u = universe
    sele1_ag = u.select_atoms(sele1_str)
    sele2_ag = u.select_atoms(sele2_str)
    sele1_set = set(sele1_ag.indices)
    sele2_set = set(sele2_ag.indices)

    N = len(u.atoms)
    all_labels = [_atom_label(a) for a in u.atoms]

    # ---- backbone mask (per-atom of whole universe) ----
    is_backbone = np.array([a.name in BACKBONE_NAMES for a in u.atoms], dtype=bool)

    # ---- ligand mask ----
    try:
        prot_idx = set(u.select_atoms("protein").indices)
        nucl_idx = set(u.select_atoms("nucleic").indices)
        try:
            water_idx = set(u.select_atoms("water or resname HOH TIP3 TIP3P SOL").indices)
        except Exception:
            water_idx = set()
        is_ligand = np.array(
            [i not in prot_idx and i not in nucl_idx and i not in water_idx
             for i in range(N)],
            dtype=bool,
        )
    except Exception:
        is_ligand = np.zeros(N, dtype=bool)

    # ---- disulfide CYS ----
    disulfide_resids: set[tuple] = set()
    try:
        for sg in u.select_atoms("resname CYS CYX and name SG"):
            nearby = u.select_atoms(
                f"resname CYS CYX and name SG and within 2.5 of index {sg.index}"
            )
            if len(nearby) >= 2:
                for a in nearby:
                    disulfide_resids.add((a.chainID, int(a.resid)))
    except Exception:
        pass

    # ===========================================================
    # AROMATIC RINGS
    # ===========================================================
    ring_idx_rows, ring_lbls, ring_m1, ring_m2 = [], [], [], []

    # --- Protein aromatic rings (getcontacts-compatible hardcoded triads) ---
    for resname, atom_names in AROMATIC_DEFS.items():
        try:
            res_group = u.select_atoms(f"resname {resname}").residues
        except Exception:
            continue
        for res in res_group:
            atom_map = {a.name: a for a in res.atoms}
            if not all(n in atom_map for n in atom_names):
                continue
            idxs = [atom_map[n].index for n in atom_names]
            ring_idx_rows.append(idxs)
            ring_lbls.append(_atom_label(atom_map[atom_names[0]]))
            ring_m1.append(any(i in sele1_set for i in idxs))
            ring_m2.append(any(i in sele2_set for i in idxs))

    # --- Nucleic acid aromatic rings (per-base proper triads) ---
    try:
        for res in u.select_atoms("nucleic").residues:
            atom_map = {a.name: a for a in res.atoms}
            for resnames_set, triads in NUCLEIC_RING_DEFS:
                if res.resname in resnames_set:
                    for triad in triads:
                        if all(n in atom_map for n in triad):
                            idxs = [atom_map[n].index for n in triad]
                            ring_idx_rows.append(idxs)
                            ring_lbls.append(_atom_label(atom_map[triad[0]]))
                            ring_m1.append(any(i in sele1_set for i in idxs))
                            ring_m2.append(any(i in sele2_set for i in idxs))
                    break
    except Exception:
        pass

    # --- Ligand/small-molecule aromatic rings (RDKit-based, optional) ---
    try:
        lig_ag = u.select_atoms(
            f"({sele1_str} or {sele2_str}) and not protein and not nucleic "
            "and not (water or resname HOH TIP3 TIP3P SOL)"
        )
        if len(lig_ag) > 0:
            _collect_rdkit_rings(
                lig_ag, sele1_set, sele2_set,
                ring_idx_rows, ring_lbls, ring_m1, ring_m2,
            )
    except Exception as e:
        pass  # RDKit not available or ligand has no recognisable rings

    ring_indices = (np.array(ring_idx_rows, dtype=np.int32)
                    if ring_idx_rows else np.empty((0, 3), dtype=np.int32))

    # ===========================================================
    # H-BOND DONORS (explicit H required)
    # ===========================================================
    d_idx, h_idx, d_lbl, d_m1, d_m2 = [], [], [], [], []
    d_is_bb, d_chain, d_resid, d_is_lig = [], [], [], []

    # Ensure bonds are available
    has_bonds = False
    try:
        if hasattr(u, "bonds") and len(u.bonds) > 0:
            has_bonds = True
    except Exception:
        pass
        
    if not has_bonds:
        try:
            u.atoms.guess_bonds()
        except Exception:
            pass

    try:
        union_ag = u.select_atoms(f"({sele1_str}) or ({sele2_str})")
        heavy_donors = union_ag.select_atoms(
            "not element H and (element N O S) and not (water or resname HOH TIP3 SOL)"
        )
        for atom in heavy_donors:
            try:
                bonded_hs = [a for a in atom.bonded_atoms if a.element == "H"]
            except Exception:
                bonded_hs = []
            for h in bonded_hs:
                d_idx.append(atom.index)
                h_idx.append(h.index)
                d_lbl.append(_atom_label(atom))
                d_m1.append(atom.index in sele1_set)
                d_m2.append(atom.index in sele2_set)
                d_is_bb.append(atom.name in BACKBONE_NAMES)
                d_chain.append(atom.chainID)
                d_resid.append(int(atom.resid))
                d_is_lig.append(is_ligand[atom.index])
    except Exception as e:
        print(f"[ultracontacts] Warning: H-bond donor detection failed: {e}")

    # ===========================================================
    # H-BOND ACCEPTORS
    # ===========================================================
    a_idx, a_lbl, a_m1, a_m2 = [], [], [], []
    a_is_bb, a_chain, a_resid, a_is_lig = [], [], [], []

    try:
        union_ag = u.select_atoms(f"({sele1_str}) or ({sele2_str})")
        acc_atoms = union_ag.select_atoms(
            "element N O F S and not element H "
            "and not (water or resname HOH TIP3 SOL)"
        )
        for atom in acc_atoms:
            a_idx.append(atom.index)
            a_lbl.append(_atom_label(atom))
            a_m1.append(atom.index in sele1_set)
            a_m2.append(atom.index in sele2_set)
            a_is_bb.append(atom.name in BACKBONE_NAMES)
            a_chain.append(atom.chainID)
            a_resid.append(int(atom.resid))
            a_is_lig.append(is_ligand[atom.index])
    except Exception as e:
        print(f"[ultracontacts] Warning: H-bond acceptor detection failed: {e}")

    # ===========================================================
    # SALT BRIDGES
    # ===========================================================
    an_idx, an_lbl, an_m1, an_m2, an_lig = [], [], [], [], []
    cat_idx, cat_lbl, cat_m1, cat_m2, cat_lig = [], [], [], [], []

    for resname, names in ANION_DEFS.items():
        for name in names:
            try:
                for atom in u.select_atoms(f"resname {resname} and name {name}"):
                    an_idx.append(atom.index)
                    an_lbl.append(_atom_label(atom))
                    an_m1.append(atom.index in sele1_set)
                    an_m2.append(atom.index in sele2_set)
                    an_lig.append(False)
            except Exception:
                pass

    try:
        for atom in u.select_atoms(f"nucleic and name {' '.join(NUCLEIC_ANION_NAMES)}"):
            an_idx.append(atom.index)
            an_lbl.append(_atom_label(atom))
            an_m1.append(atom.index in sele1_set)
            an_m2.append(atom.index in sele2_set)
            an_lig.append(False)
    except Exception:
        pass

    for resname, names in CATION_DEFS.items():
        for name in names:
            try:
                for atom in u.select_atoms(f"resname {resname} and name {name}"):
                    cat_idx.append(atom.index)
                    cat_lbl.append(_atom_label(atom))
                    cat_m1.append(atom.index in sele1_set)
                    cat_m2.append(atom.index in sele2_set)
                    cat_lig.append(False)
            except Exception:
                pass

    # Simple ligand ion detection: O in carboxylate pattern (2 O bonded to sp2 C)
    try:
        lig_atoms = u.select_atoms(
            f"({sele1_str} or {sele2_str}) and not protein and not nucleic "
            "and not (water or resname HOH TIP3 SOL)"
        )
        _detect_ligand_ions(lig_atoms, is_ligand, sele1_set, sele2_set,
                            an_idx, an_lbl, an_m1, an_m2, an_lig,
                            cat_idx, cat_lbl, cat_m1, cat_m2, cat_lig)
    except Exception as e:
        print(f"[ultracontacts] Warning: ligand ion detection failed: {e}")

    # ===========================================================
    # HYDROPHOBICS
    # ===========================================================
    hp_idx_, hp_lbl_, hp_m1_, hp_m2_ = [], [], [], []
    hp_chain_, hp_resid_, hp_is_bb_, hp_radii_ = [], [], [], []

    try:
        hp_sel_str = " or ".join(f"resname {r}" for r in HYDROPHOBIC_RESNAMES)
        all_union = u.select_atoms(f"({sele1_str}) or ({sele2_str})")
        hp_atoms = all_union.select_atoms(f"({hp_sel_str}) and element C S")
        for atom in hp_atoms:
            if atom.resname in ("CYS", "CYX") and (atom.chainID, int(atom.resid)) in disulfide_resids:
                continue
            hp_idx_.append(atom.index)
            hp_lbl_.append(_atom_label(atom))
            hp_m1_.append(atom.index in sele1_set)
            hp_m2_.append(atom.index in sele2_set)
            hp_chain_.append(atom.chainID)
            hp_resid_.append(int(atom.resid))
            hp_is_bb_.append(atom.name in BACKBONE_NAMES)
            hp_radii_.append(_get_vdw(atom.element))
    except Exception as e:
        print(f"[ultracontacts] Warning: hydrophobic group detection failed: {e}")

    # ===========================================================
    # VDW (all heavy atoms in sele1 / sele2)
    # ===========================================================
    def _build_vdw(ag):
        idx_, lbl_, rad_, ch_, rid_, lig_, bb_ = [], [], [], [], [], [], []
        try:
            heavy = ag.select_atoms("not element H")
        except Exception:
            heavy = ag
        for atom in heavy:
            idx_.append(atom.index)
            lbl_.append(_atom_label(atom))
            rad_.append(_get_vdw(atom.element))
            ch_.append(atom.chainID)
            rid_.append(int(atom.resid))
            lig_.append(bool(is_ligand[atom.index]))
            bb_.append(atom.name in BACKBONE_NAMES)
        return idx_, lbl_, rad_, ch_, rid_, lig_, bb_

    v1 = _build_vdw(sele1_ag)
    v2 = _build_vdw(sele2_ag)

    # ===========================================================
    # Assemble
    # ===========================================================
    return ChemicalGroups(
        n_atoms=N,
        # rings
        ring_indices=ring_indices,
        ring_labels=ring_lbls,
        ring_in_sele1=_to_bool(ring_m1),
        ring_in_sele2=_to_bool(ring_m2),
        # H-bond donors
        donor_indices=_to_i32(d_idx),
        hydrogen_indices=_to_i32(h_idx),
        donor_labels=d_lbl,
        donor_in_sele1=_to_bool(d_m1),
        donor_in_sele2=_to_bool(d_m2),
        donor_is_bb=_to_bool(d_is_bb),
        donor_chain=np.array(d_chain, dtype=object),
        donor_resid=np.array(d_resid, dtype=np.int32) if d_resid else np.empty(0, dtype=np.int32),
        donor_is_ligand=_to_bool(d_is_lig),
        # H-bond acceptors
        acceptor_indices=_to_i32(a_idx),
        acceptor_labels=a_lbl,
        acceptor_in_sele1=_to_bool(a_m1),
        acceptor_in_sele2=_to_bool(a_m2),
        acceptor_is_bb=_to_bool(a_is_bb),
        acceptor_chain=np.array(a_chain, dtype=object),
        acceptor_resid=np.array(a_resid, dtype=np.int32) if a_resid else np.empty(0, dtype=np.int32),
        acceptor_is_ligand=_to_bool(a_is_lig),
        # salt bridges
        anion_indices=_to_i32(an_idx),
        anion_labels=an_lbl,
        anion_in_sele1=_to_bool(an_m1),
        anion_in_sele2=_to_bool(an_m2),
        anion_is_ligand=_to_bool(an_lig),
        cation_indices=_to_i32(cat_idx),
        cation_labels=cat_lbl,
        cation_in_sele1=_to_bool(cat_m1),
        cation_in_sele2=_to_bool(cat_m2),
        cation_is_ligand=_to_bool(cat_lig),
        # hydrophobics
        hp_indices=_to_i32(hp_idx_),
        hp_labels=hp_lbl_,
        hp_in_sele1=_to_bool(hp_m1_),
        hp_in_sele2=_to_bool(hp_m2_),
        hp_chain=np.array(hp_chain_, dtype=object),
        hp_resid=np.array(hp_resid_, dtype=np.int32) if hp_resid_ else np.empty(0, dtype=np.int32),
        hp_is_bb=_to_bool(hp_is_bb_),
        hp_radii=np.array(hp_radii_, dtype=np.float32) if hp_radii_ else np.empty(0, dtype=np.float32),
        # vdw
        vdw_indices1=_to_i32(v1[0]),
        vdw_labels1=v1[1],
        vdw_radii1=np.array(v1[2], dtype=np.float32) if v1[2] else np.empty(0, dtype=np.float32),
        vdw_chain1=np.array(v1[3], dtype=object),
        vdw_resid1=np.array(v1[4], dtype=np.int32) if v1[4] else np.empty(0, dtype=np.int32),
        vdw_is_ligand1=_to_bool(v1[5]),
        vdw_is_bb1=_to_bool(v1[6]),
        vdw_indices2=_to_i32(v2[0]),
        vdw_labels2=v2[1],
        vdw_radii2=np.array(v2[2], dtype=np.float32) if v2[2] else np.empty(0, dtype=np.float32),
        vdw_chain2=np.array(v2[3], dtype=object),
        vdw_resid2=np.array(v2[4], dtype=np.int32) if v2[4] else np.empty(0, dtype=np.int32),
        vdw_is_ligand2=_to_bool(v2[5]),
        vdw_is_bb2=_to_bool(v2[6]),
        # global
        is_backbone=is_backbone,
        is_ligand=is_ligand,
        all_labels=all_labels,
        disulfide_resids=disulfide_resids,
    )


# ---------------------------------------------------------------------------
# Ligand ion detection (simplified carboxylate / ammonium detection)
# ---------------------------------------------------------------------------
def _detect_ligand_ions(lig_atoms, is_ligand, sele1_set, sele2_set,
                         an_idx, an_lbl, an_m1, an_m2, an_lig,
                         cat_idx, cat_lbl, cat_m1, cat_m2, cat_lig):
    """
    Detect carboxylate-like anions and quaternary/aromatic N cations in ligands.
    Requires bond information to be present.
    """
    for atom in lig_atoms:
        if not is_ligand[atom.index]:
            continue
        try:
            bonded = list(atom.bonded_atoms)
        except Exception:
            continue

        # sp2 C bonded to exactly two O's → carboxylate anion; add the oxygens
        if atom.element == "C" and len(bonded) == 3:
            oxygens = [a for a in bonded if a.element == "O"]
            carbons = [a for a in bonded if a.element == "C"]
            if len(oxygens) == 2 and len(carbons) == 1:
                for o_atom in oxygens:
                    an_idx.append(o_atom.index)
                    an_lbl.append(_atom_label(o_atom))
                    an_m1.append(o_atom.index in sele1_set)
                    an_m2.append(o_atom.index in sele2_set)
                    an_lig.append(True)

        # N with 4 bonds → quaternary ammonium cation
        if atom.element == "N" and len(bonded) >= 3:
            # Count non-H bonds
            non_h = [a for a in bonded if a.element != "H"]
            if len(non_h) >= 3:
                cat_idx.append(atom.index)
                cat_lbl.append(_atom_label(atom))
                cat_m1.append(atom.index in sele1_set)
                cat_m2.append(atom.index in sele2_set)
                cat_lig.append(True)


# ---------------------------------------------------------------------------
# Pre-compute cross-selection validity mask (CPU, once)
# ---------------------------------------------------------------------------
def build_dual_sele_mask(m1_a: np.ndarray, m2_a: np.ndarray,
                          m1_b: np.ndarray, m2_b: np.ndarray) -> np.ndarray:
    """
    Returns bool (A, B) where True means the pair (a, b) crosses sele1/sele2.
    Valid if (a in sele1 and b in sele2) or (a in sele2 and b in sele1).
    """
    return (np.outer(m1_a, m2_b) | np.outer(m2_a, m1_b))


def build_residue_diff_mask(chain_a: np.ndarray, resid_a: np.ndarray,
                             chain_b: np.ndarray, resid_b: np.ndarray,
                             min_diff: int, bb_a: np.ndarray | None = None,
                             bb_b: np.ndarray | None = None,
                             adjacent_bb_only: bool = False) -> np.ndarray:
    """
    Returns bool (A, B) where True means the pair is allowed (not excluded).

    - Different chains: always allowed
    - Same chain, |resid_diff| >= min_diff: allowed
    - Same chain, |resid_diff| < min_diff: excluded (unless filtered by backbone)

    If bb_a/bb_b provided and adjacent_bb_only=False: only filters when BOTH
    atoms are backbone (original behaviour for H-bonds).

    If adjacent_bb_only=True: same-residue (diff=0) always excluded;
    adjacent residues (diff=1, same chain) excluded only when BOTH atoms are
    backbone.  This allows sidechain contacts between adjacent residues while
    filtering out covalently bonded backbone atoms.
    """
    same_chain = (chain_a[:, None] == chain_b[None, :])          # (A, B)
    res_diff = np.abs(resid_a[:, None].astype(int) -
                      resid_b[None, :].astype(int))               # (A, B)

    if adjacent_bb_only and bb_a is not None and bb_b is not None:
        # Same residue: always excluded
        same_res = same_chain & (res_diff == 0)
        # Adjacent residue: only exclude backbone-backbone
        adjacent = same_chain & (res_diff == 1)
        both_bb = bb_a[:, None] & bb_b[None, :]
        too_close = same_res | (adjacent & both_bb)
    else:
        too_close = same_chain & (res_diff < min_diff)             # (A, B)
        if bb_a is not None and bb_b is not None:
            both_bb = bb_a[:, None] & bb_b[None, :]
            too_close = too_close & both_bb

    return ~too_close
