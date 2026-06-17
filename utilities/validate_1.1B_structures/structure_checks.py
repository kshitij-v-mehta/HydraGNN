"""
structure_checks.py
-------------------
Validation and synthesizability checks for atomistic structures.

Designed to be imported by validate_structures.py. Can also be used
standalone for interactive inspection or unit testing.

Public API
----------
check_stage1(cx, cy, cz, fx, fy, fz, en, N, args) -> (bool, str, float)
    Physical validity gate. Returns (is_valid, failure_reason, min_dist).

check_stage2(cx, cy, cz, at, N) -> (bool, str)
    Synthesizability gate. Returns (is_synthesizable, failure_reason).
    Only call on structures that passed Stage 1.

Chemistry tables (importable directly if needed)
-------------------------------------------------
COVALENT_RADII   : dict[int, float]  — Alvarez 2008 covalent radii in Å
MAX_VALENCE      : dict[int, int]    — max bonds per element
SYNTHESIZABLE_Z  : set[int]          — whitelisted atomic numbers
BOND_TOLERANCE   : float             — extra Å added to covalent radii sum
MIN_BOND_DIST    : float             — minimum distance to count as a bond
"""

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except ImportError:
    _HAVE_KDTREE = False


# ---------------------------------------------------------------------------
# Chemistry tables
# ---------------------------------------------------------------------------

# Covalent radii in Angstroms (Alvarez 2008), indexed by atomic number 1..83.
# Used to compute per-pair maximum bond distance: r_i + r_j + BOND_TOLERANCE.
# Source: DOI 10.1039/b801115j
COVALENT_RADII = {
     1: 0.31,  2: 0.28,  3: 1.28,  4: 0.96,  5: 0.84,  6: 0.76,  7: 0.71,
     8: 0.66,  9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11,
    15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70,
    22: 1.60, 23: 1.53, 24: 1.39, 25: 1.61, 26: 1.52, 27: 1.50, 28: 1.24,
    29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20,
    36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54,
    43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42,
    50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15,
    57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98,
    64: 1.96, 65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87,
    71: 1.87, 72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41,
    78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48,
}

# Maximum number of bonds per element (most common stable oxidation states).
MAX_VALENCE = {
     1: 1,   2: 0,   3: 1,   4: 2,   5: 3,   6: 4,   7: 4,   8: 2,
     9: 1,  10: 0,  11: 1,  12: 2,  13: 3,  14: 4,  15: 5,  16: 6,
    17: 7,  18: 0,  19: 1,  20: 2,  21: 3,  22: 4,  23: 5,  24: 6,
    25: 7,  26: 6,  27: 5,  28: 4,  29: 4,  30: 2,  31: 3,  32: 4,
    33: 5,  34: 6,  35: 7,  36: 2,  37: 1,  38: 2,  39: 3,  40: 4,
    41: 5,  42: 6,  43: 7,  44: 8,  45: 6,  46: 4,  47: 3,  48: 2,
    49: 3,  50: 4,  51: 5,  52: 6,  53: 7,  54: 4,  55: 1,  56: 2,
    57: 3,  58: 4,  59: 4,  60: 3,  61: 3,  62: 3,  63: 3,  64: 3,
    65: 4,  66: 3,  67: 3,  68: 3,  69: 3,  70: 3,  71: 3,  72: 4,
    73: 5,  74: 6,  75: 7,  76: 8,  77: 6,  78: 4,  79: 3,  80: 2,
    81: 3,  82: 4,  83: 5,
}

# Elements with no stable isotopes — excluded from synthesizable set.
# Tc (43): no stable isotopes, no natural occurrence.
# Pm (61): no stable isotopes, no natural occurrence.
_EXCLUDED_Z = {43, 61}

# Full synthesizable whitelist: Z 1..83 minus unstable elements.
SYNTHESIZABLE_Z = set(range(1, 84)) - _EXCLUDED_Z

# Extra Å added to the sum of covalent radii to define the bond cutoff.
# Covers ionic and metallic bonds, which are longer than covalent bonds.
BOND_TOLERANCE = 0.40

# Distance below which two atoms are considered overlapping, not bonded.
MIN_BOND_DIST = 0.50

# Elements for which an odd bond count indicates a radical (C, N, O, H).
_RADICAL_CHECK_Z = {1, 6, 7, 8}

# Pre-computed global neighbor search cutoff (max possible bond distance).
_GLOBAL_BOND_CUTOFF = max(COVALENT_RADII.values()) * 2 + BOND_TOLERANCE  # ~3.3 Å


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bond_cutoff(z1: int, z2: int) -> float:
    """Maximum distance (Å) considered a bond between elements z1 and z2."""
    r1 = COVALENT_RADII.get(z1, 1.5)
    r2 = COVALENT_RADII.get(z2, 1.5)
    return r1 + r2 + BOND_TOLERANCE


def _compute_bond_counts(coords: np.ndarray, atom_types: np.ndarray) -> np.ndarray:
    """
    Count bonds per atom using element-specific covalent radii cutoffs.

    Parameters
    ----------
    coords     : (N, 3) float64 array of atomic positions in Å
    atom_types : (N,)   int32   array of atomic numbers

    Returns
    -------
    bond_counts : (N,) int32 array — number of bonds for each atom
    """
    N = len(coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=_GLOBAL_BOND_CUTOFF, output_type='ndarray')

    bond_counts = np.zeros(N, dtype=np.int32)
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < MIN_BOND_DIST:
            continue
        if dist <= _bond_cutoff(int(atom_types[i]), int(atom_types[j])):
            bond_counts[i] += 1
            bond_counts[j] += 1

    return bond_counts


# ---------------------------------------------------------------------------
# Stage 1: physical validity checks
# ---------------------------------------------------------------------------

def check_energy(e_val: float, energy_lo: float, energy_hi: float) -> tuple[bool, str]:
    """
    Check formation energy is finite and within [energy_lo, energy_hi] eV.

    Returns (passed, reason). reason is '' if passed.
    """
    if not np.isfinite(e_val):
        return False, "energy_nan_inf"
    if e_val < energy_lo or e_val > energy_hi:
        return False, "energy_out_of_range"
    return True, ""


def check_forces(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
                 max_force: float) -> tuple[bool, str]:
    """
    Check all force components are finite and |F| <= max_force eV/Å.

    Returns (passed, reason).
    """
    f_mag_sq = fx**2 + fy**2 + fz**2
    if not np.all(np.isfinite(f_mag_sq)):
        return False, "force_nan_inf"
    if f_mag_sq.max() > max_force**2:
        return False, "force_too_large"
    return True, ""


def check_min_distance(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray,
                       min_dist: float) -> tuple[bool, str, float]:
    """
    Check no two atoms are closer than min_dist Å.

    Returns (passed, reason, min_dist_found).
    min_dist_found is -1.0 if N < 2 (check skipped).
    Requires scipy.
    """
    if not _HAVE_KDTREE:
        return True, "", -1.0

    N = len(cx)
    if N < 2:
        return True, "", -1.0

    coords = np.stack([cx, cy, cz], axis=1)
    tree   = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    min_d  = float(dists[:, 1].min())

    if min_d < min_dist:
        return False, "atom_overlap", min_d
    return True, "", min_d


def check_stage1(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray,
                 fx: np.ndarray, fy: np.ndarray, fz: np.ndarray,
                 en: float, N: int,
                 energy_lo: float, energy_hi: float,
                 max_force: float, min_dist: float,
                 do_mindist: bool = True) -> tuple[bool, str, float]:
    """
    Run all Stage 1 physical validity checks in fast-to-slow order.

    Parameters
    ----------
    cx, cy, cz   : per-atom coordinate arrays (Å)
    fx, fy, fz   : per-atom force arrays (eV/Å)
    en           : formation energy scalar (eV)
    N            : number of atoms
    energy_lo/hi : energy range (eV)
    max_force    : max allowed force magnitude (eV/Å)
    min_dist     : min allowed interatomic distance (Å)
    do_mindist   : whether to run the distance check

    Returns
    -------
    (passed, failure_reason, min_dist_found)
    passed         : True if all checks passed
    failure_reason : empty string if passed, else one of:
                     'energy_nan_inf', 'energy_out_of_range',
                     'force_nan_inf', 'force_too_large', 'atom_overlap'
    min_dist_found : minimum interatomic distance found, or -1.0 if not computed
    """
    # 1a: energy
    ok, reason = check_energy(en, energy_lo, energy_hi)
    if not ok:
        return False, reason, -1.0

    # 1b: forces
    ok, reason = check_forces(fx, fy, fz, max_force)
    if not ok:
        return False, reason, -1.0

    # 1c+1d: min distance
    if do_mindist:
        ok, reason, min_d = check_min_distance(cx, cy, cz, min_dist)
        if not ok:
            return False, reason, min_d
        return True, "", min_d

    return True, "", -1.0


# ---------------------------------------------------------------------------
# Stage 2: synthesizability checks
# ---------------------------------------------------------------------------

def check_element_whitelist(atom_types: np.ndarray) -> tuple[bool, str]:
    """
    Check all elements are in the synthesizable whitelist (Z 1..83, no Tc/Pm).

    Returns (passed, reason).
    """
    unique_z = set(atom_types.tolist())
    bad = unique_z - SYNTHESIZABLE_Z
    if bad:
        return False, f"non_synthesizable_elements:{sorted(bad)}"
    return True, ""


def check_bonding(coords: np.ndarray,
                  atom_types: np.ndarray) -> tuple[bool, str, np.ndarray]:
    """
    Check bonding environment for isolation, valence violations, and radicals.

    Parameters
    ----------
    coords     : (N, 3) float64 array of positions in Å
    atom_types : (N,)   int32   array of atomic numbers

    Returns
    -------
    (passed, failure_reason, bond_counts)
    failure_reason is one of: '', 'isolated_atom', 'valence_exceeded', 'radical'
    bond_counts is the per-atom bond count array (useful for diagnostics)
    """
    bond_counts = _compute_bond_counts(coords, atom_types)

    # Check for isolated atoms
    if np.any(bond_counts == 0):
        return False, "isolated_atom", bond_counts

    # Check valence and radicals per atom
    for i in range(len(atom_types)):
        zi = int(atom_types[i])
        bc = int(bond_counts[i])
        mv = MAX_VALENCE.get(zi, 6)

        if bc > mv:
            return False, "valence_exceeded", bond_counts

        if zi in _RADICAL_CHECK_Z and bc % 2 != 0:
            return False, "radical", bond_counts

    return True, "", bond_counts


def check_stage2(cx: np.ndarray, cy: np.ndarray, cz: np.ndarray,
                 atom_types: np.ndarray,
                 N: int) -> tuple[bool, str]:
    """
    Run all Stage 2 synthesizability checks in fast-to-slow order.
    Only call on structures that have already passed Stage 1.

    Parameters
    ----------
    cx, cy, cz  : per-atom coordinate arrays (Å)
    atom_types  : per-atom atomic number array
    N           : number of atoms

    Returns
    -------
    (passed, failure_reason)
    failure_reason is one of:
        '', 'non_synthesizable_elements:...', 'isolated_atom',
        'valence_exceeded', 'radical'
    """
    if not _HAVE_KDTREE:
        # Cannot run bonding checks without scipy; pass element check only
        return check_element_whitelist(atom_types)

    # 2a: element whitelist (cheap — set operation)
    ok, reason = check_element_whitelist(atom_types)
    if not ok:
        return False, reason

    # 2b-2d: bonding checks (requires cKDTree)
    if N >= 2:
        coords = np.stack([cx, cy, cz], axis=1)
        ok, reason, _ = check_bonding(coords, atom_types)
        if not ok:
            return False, reason

    return True, ""
