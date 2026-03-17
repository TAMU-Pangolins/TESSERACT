"""Helpers for reading AME masses and computing reaction Q-values."""

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

AMU_TO_KG = 1.66053906660e-27
KEV_PER_AMU = 931_494.10242
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_AME_PATH = DATA_DIR / "ame20.csv"
ELEMENT_SYMBOLS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}
SPECIAL_NUCLIDES = {
    "n": (0, 1),
    "p": (1, 1),
}


_NUMERIC_RE = re.compile(r"[^\dEe+\-\.]")


def _clean_numeric(value: str) -> float:
    cleaned = _NUMERIC_RE.sub("", value.strip())
    if cleaned in ("", "+", "-"):
        raise ValueError(f"Cannot parse numeric value from {value!r}")
    return float(cleaned)


@lru_cache(maxsize=1)
def load_ame(path: str | Path = DEFAULT_AME_PATH) -> Dict[Tuple[int, int], float]:
    """
    Load AME mass excess values keyed by `(Z, A)`.

    Parameters
    ----------
    path : str or Path, default=DEFAULT_AME_PATH
        CSV file containing AME-derived masses and mass excesses.

    Returns
    -------
    dict[tuple[int, int], float]
        Mass excess lookup table in keV.
    """
    lut = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lut[(int(row["Z"]), int(row["A"]))] = _clean_numeric(row["mass_excess"])
    return lut


def calc_qval(Z_p, A_p, Z_t, A_t, Z_eject, A_eject, ame=None):
    """
    Compute the reaction Q-value in MeV from AME mass excesses.

    Parameters
    ----------
    Z_p, A_p : int
        Projectile proton and mass numbers.
    Z_t, A_t : int
        Target proton and mass numbers.
    Z_eject, A_eject : int
        Ejectile proton and mass numbers.
    ame : dict or None, optional
        Preloaded AME lookup table.

    Returns
    -------
    float or str
        Q-value in MeV, or a human-readable error string when the requested
        reaction cannot be resolved from the available masses.
    """
    ame = ame or load_ame()
    Z_res = Z_p + Z_t - Z_eject
    A_res = A_p + A_t - A_eject
    if Z_res < 0 or A_res < 0:
        return "The reaction does not make sense. The Z and A of the residual nuclei is negative!"
    try:
        return (
            ame[(Z_p, A_p)]
            + ame[(Z_t, A_t)]
            - ame[(Z_eject, A_eject)]
            - ame[(Z_res, A_res)]
        ) * 1e-3
    except KeyError:
        return "Mass excess missing for one of the nuclides."


def atomic_mass_u(Z: int, A: int, ame=None) -> float:
    """
    Return the atomic mass (u) for a nucleus using its mass excess.

    Raises
    ------
    KeyError
        If the AME table does not contain the requested nuclide.
    """
    ame = ame or load_ame()
    try:
        mass_excess_keV = ame[(Z, A)]
    except KeyError as exc:
        raise KeyError(f"Mass excess missing for Z={Z}, A={A}") from exc
    return A + mass_excess_keV / KEV_PER_AMU


def atomic_mass_kg(Z: int, A: int, ame=None) -> float:
    """
    Atomic mass converted to kilograms.

    Parameters
    ----------
    Z, A : int
        Proton and mass numbers.
    ame : dict or None, optional
        Preloaded AME lookup table.
    """
    return atomic_mass_u(Z, A, ame=ame) * AMU_TO_KG


def split_nuclide_token(token: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Split a token like `22Mg` into `(22, "Mg")`.

    Returns `(None, None)` when the token cannot be parsed.
    """
    if token is None:
        return None, None
    match = re.match(r"^\s*(\d+)?([A-Za-z]+)", token.strip())
    if not match:
        return None, None
    mass = match.group(1)
    symbol = match.group(2)
    return (int(mass) if mass is not None else None, symbol.capitalize())


def element_symbol_to_Z(symbol: Optional[str]) -> Optional[int]:
    """Map an element symbol or special light-particle token to its proton number."""
    if symbol is None:
        return None
    lower = symbol.lower()
    if lower in SPECIAL_NUCLIDES:
        return SPECIAL_NUCLIDES[lower][0]
    return ELEMENT_SYMBOLS.get(symbol.capitalize())


def interpret_z_token(token: Optional[str]) -> Optional[int]:
    """
    Interpret a proton-number token given as either numeric text or nuclide text.

    Returns `None` when the token cannot be resolved.
    """
    if token is None:
        return None
    try:
        return int(float(token))
    except ValueError:
        mass, symbol = split_nuclide_token(token)
        if symbol:
            return element_symbol_to_Z(symbol)
        return None


def mass_from_token(
    token: Optional[str], z_hint: Optional[int], ame=None
) -> Optional[float]:
    """
    Resolve a token to a mass in kg using literal values or the AME table.

    Parameters
    ----------
    token : str or None
        Literal mass, nuclide token, or special particle token.
    z_hint : int or None
        Optional proton-number hint used when `token` lacks an element symbol.
    ame : dict or None, optional
        Preloaded AME lookup table.

    Returns
    -------
    float or None
        Mass in kg when the token can be resolved, else `None`.
    """
    if token is None:
        return None
    try:
        mass_amu = float(token)
        return mass_amu * AMU_TO_KG
    except ValueError:
        pass
    mass_number, symbol = split_nuclide_token(token)
    z_val = z_hint
    if symbol:
        sym_z = element_symbol_to_Z(symbol)
        if sym_z is not None:
            z_val = sym_z
    if z_val is None:
        return None
    if mass_number is None and symbol:
        special = SPECIAL_NUCLIDES.get(symbol.lower())
        if special:
            mass_number = special[1]
    if mass_number is None:
        return None
    return atomic_mass_kg(z_val, mass_number, ame=ame)


# e.g. Example Q-value calculation for 7Li + 48Ca -> 54V + n
# print(calc_qval(3,7,20,48,0,1))
