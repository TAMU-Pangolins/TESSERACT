from __future__ import annotations

from pathlib import Path

import numpy as np

from common.densities_retrieval import (
    J_index,
    apply_hfb_corrections,
    build_rho_interpolator,
    read_hfb_cor,
    read_hfb_tab,
)

from .config import resolve_data_root


def resolve_density_paths(
    Z: int,
    data_root: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """
    Resolve canonical paths for level-density files of proton number Z.
    Returns (tab_path, cor_path_or_None), where files are strictly:
      zXXX.tab (required), zXXX.cor (optional)
    """
    if data_root is None:
        root = resolve_data_root()
    else:
        root = Path(data_root)
    tab = root / f"z{Z:03d}.tab"
    if not tab.exists():
        raise FileNotFoundError(f"Missing level-density .tab file: {tab}")
    cor = root / f"z{Z:03d}.cor"
    return tab, (cor if cor.exists() else None)


def load_rho_function(
    tab_path, cor_path=None, *, use_corrections=False, warn_if_ignored=True
):
    """
    Load rho(U,J,pi) from a zXXX.tab file.
    !! Set use_corrections=True to re-enable applying .cor !!
    """
    rec = read_hfb_tab(tab_path)
    if use_corrections and cor_path:
        cor = read_hfb_cor(cor_path)
        rec = apply_hfb_corrections(rec, cor)
    elif cor_path and warn_if_ignored:
        # Soft notice; flip use_corrections=True later to re-enable.
        print(
            f"[info] Ignoring corrections file for now: {cor_path} (set use_corrections=True to apply)"
        )
    return build_rho_interpolator(rec)


def load_hfb_record(
    tab_path, cor_path=None, *, use_corrections=False, warn_if_ignored=True
):
    """
    Load the full HFB record (including spin-resolved rho_J) from a zXXX.tab file.
    """
    rec = read_hfb_tab(tab_path)
    if use_corrections and cor_path:
        cor = read_hfb_cor(cor_path)
        rec = apply_hfb_corrections(rec, cor)
    elif cor_path and warn_if_ignored:
        print(
            f"[info] Ignoring corrections file for now: {cor_path} (set use_corrections=True to apply)"
        )
    return rec


def rho_levels_per_eV_from_E(rho_UJpi, A, J_phys, pi, U_of_E_mev):
    """
    Returns rho(E) in levels/eV for fixed (J_phys, pi). See densities_retrieval.J_index/spin_grid.
    """
    Jcol = J_index(J_phys, A)

    def rho_E(E_eV):
        U = np.asarray(U_of_E_mev(np.asarray(E_eV, dtype=float)), dtype=float)
        rho_mev = rho_UJpi(U, Jcol, pi)  # levels / MeV
        return np.asarray(rho_mev, dtype=float) * 1e-6  # -> levels / eV

    return rho_E


def rho_total_levels_per_eV_from_E(record, pi, U_of_E_mev):
    """
    Returns total rho(E) in levels/eV for fixed parity (sum over spins).
    """
    block = record.positive if pi == +1 else record.negative
    U_grid = block.U
    rho_level = block.Rho_level

    def rho_E(E_eV):
        U = np.asarray(U_of_E_mev(np.asarray(E_eV, dtype=float)), dtype=float)
        rho_mev = np.interp(
            U, U_grid, rho_level, left=rho_level[0], right=rho_level[-1]
        )
        return np.asarray(rho_mev, dtype=float) * 1e-6

    return rho_E


def build_density_grid(
    *,
    # choose by Z (preferred) or override with explicit tab_path
    Z: int | None = None,
    data_root: str | Path | None = None,
    tab_path: str | Path | None = None,
    # physics/indexing knobs used by your existing adapter
    A: int = 24,
    J_phys: float = 1.0,
    pi: int = 1,
    # energy grid
    E_min_mev: float = 0.1,
    E_max_mev: float = 2.0,
    n_points: int = 2001,
    # mapping from lab energy (eV) to excitation U (MeV)
    U_of_E_mev=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (E_mev, rho_per_mev) on a uniform MeV grid.
    """
    # lazy import of your already-defined helpers in this module
    from .hfb_adapter import (
        load_rho_function,
        resolve_density_paths,
        rho_levels_per_eV_from_E,
    )

    # Resolve files
    if tab_path is None:
        if Z is None:
            raise ValueError("build_density_grid needs either tab_path=... or Z=...")
        tab_p, cor_p = resolve_density_paths(Z, data_root=data_root)
    else:
        tab_p = Path(tab_path)
        if not tab_p.exists():
            raise FileNotFoundError(f"tab_path does not exist: {tab_p}")
        # try canonical .cor next to it
        candidate_cor = tab_p.with_suffix(".cor")
        cor_p = candidate_cor if candidate_cor.exists() else None

    # Default U≈E if not provided
    if U_of_E_mev is None:
        U_of_E_mev = lambda E_eV: (np.asarray(E_eV, dtype=float) * 1e-6)

    # Let load_rho_function decide how to handle corrections when cor_p is present
    rho_UJpi = load_rho_function(
        tab_path=str(tab_p),
        cor_path=(str(cor_p) if cor_p is not None else None),
        # no external flags; hfb_adapter decides how/when to apply .cor
    )

    rho_E_eV_fn = rho_levels_per_eV_from_E(
        rho_UJpi, A=A, J_phys=J_phys, pi=pi, U_of_E_mev=U_of_E_mev
    )

    # Build grid in MeV, evaluate ρ (levels/eV) then convert to levels/MeV
    E_mev = np.linspace(float(E_min_mev), float(E_max_mev), int(n_points))
    rho_per_eV = np.asarray(rho_E_eV_fn(E_mev * 1e6), dtype=float)
    rho_per_mev = np.clip(rho_per_eV * 1e6, a_min=0.0, a_max=None)
    return E_mev, rho_per_mev


def build_total_density_grid(
    *,
    Z: int | None = None,
    data_root: str | Path | None = None,
    tab_path: str | Path | None = None,
    pi: int = 1,
    E_min_mev: float = 0.1,
    E_max_mev: float = 2.0,
    n_points: int = 2001,
    U_of_E_mev=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (E_mev, rho_per_mev) using total level density for fixed parity.
    """
    from .hfb_adapter import (
        load_hfb_record,
        resolve_density_paths,
        rho_total_levels_per_eV_from_E,
    )

    if tab_path is None:
        if Z is None:
            raise ValueError(
                "build_total_density_grid needs either tab_path=... or Z=..."
            )
        tab_p, cor_p = resolve_density_paths(Z, data_root=data_root)
    else:
        tab_p = Path(tab_path)
        if not tab_p.exists():
            raise FileNotFoundError(f"tab_path does not exist: {tab_p}")
        candidate_cor = tab_p.with_suffix(".cor")
        cor_p = candidate_cor if candidate_cor.exists() else None

    if U_of_E_mev is None:
        U_of_E_mev = lambda E_eV: (np.asarray(E_eV, dtype=float) * 1e-6)

    record = load_hfb_record(
        tab_path=str(tab_p),
        cor_path=(str(cor_p) if cor_p is not None else None),
    )

    rho_E_eV_fn = rho_total_levels_per_eV_from_E(record, pi=pi, U_of_E_mev=U_of_E_mev)

    E_mev = np.linspace(float(E_min_mev), float(E_max_mev), int(n_points))
    rho_per_eV = np.asarray(rho_E_eV_fn(E_mev * 1e6), dtype=float)
    rho_per_mev = np.clip(rho_per_eV * 1e6, a_min=0.0, a_max=None)
    return E_mev, rho_per_mev
