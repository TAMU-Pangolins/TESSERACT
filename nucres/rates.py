from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .physics import (
    AVOGADRO,
    BARNS_TO_M2,
    BOLTZMANN,
    EV_TO_J,
    MASS_PROTON,
    PI,
    reduced_mass,
)

_ENERGY_UNIT_TO_EV = {
    "ev": 1.0,
    "kev": 1e3,
    "mev": 1e6,
    "gev": 1e9,
}

_SIGMA_UNIT_TO_M2 = {
    "barn": BARNS_TO_M2,
    "b": BARNS_TO_M2,
    "m2": 1.0,
    "m^2": 1.0,
    "cm2": 1e-4,
    "cm^2": 1e-4,
}

_TEMP_UNIT_TO_K = {
    "k": 1.0,
    "kelvin": 1.0,
    "mk": 1e6,
    "gk": 1e9,
    "t9": 1e9,
}

_RATE_UNIT_SCALE = {
    "m3/mol/s": 1.0,
    "m^3/mol/s": 1.0,
    "cm3/mol/s": 1e6,
    "cm^3/mol/s": 1e6,
}


@dataclass(frozen=True)
class ReactionRateResult:
    temperature: np.ndarray
    na_sigma_v: np.ndarray
    temperature_unit: str
    rate_unit: str

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.temperature.copy(), self.na_sigma_v.copy()


def _norm_key(value: str) -> str:
    return value.replace(" ", "").lower()


def _energy_to_joules(E: Sequence[float], *, unit: str) -> np.ndarray:
    key = _norm_key(unit)
    scale = _ENERGY_UNIT_TO_EV.get(key)
    if scale is None:
        raise ValueError(f"Unsupported energy unit '{unit}'.")
    return np.asarray(E, dtype=float) * scale * EV_TO_J


def _sigma_to_m2(sigma: Sequence[float], *, unit: str) -> np.ndarray:
    key = _norm_key(unit)
    scale = _SIGMA_UNIT_TO_M2.get(key)
    if scale is None:
        raise ValueError(f"Unsupported cross section unit '{unit}'.")
    return np.asarray(sigma, dtype=float) * scale


def _temperature_to_kelvin(T: Sequence[float] | float, *, unit: str) -> np.ndarray:
    key = _norm_key(unit)
    scale = _TEMP_UNIT_TO_K.get(key)
    if scale is None:
        raise ValueError(f"Unsupported temperature unit '{unit}'.")
    return np.asarray(T, dtype=float) * scale


def na_sigma_v_from_sigma(
    E,
    sigma,
    T,
    *,
    m1: float = MASS_PROTON,
    m2: float = MASS_PROTON,
    energy_unit: str = "eV",
    sigma_unit: str = "barn",
    temperature_unit: str = "K",
    result_unit: str = "cm^3/mol/s",
    method: str = "trapz",
) -> ReactionRateResult:
    """
    Integrate a cross section σ(E) over a Maxwell–Boltzmann distribution to obtain
    N_A⟨σv⟩(T).

    Parameters
    ----------
    E : array-like
        Sampling energies corresponding to `sigma`.
    sigma : array-like
        Cross section values sampled at `E`.
    T : float or array-like
        Temperatures where the rate is evaluated.
    m1, m2 : float
        Projectile and target masses in kg.
    energy_unit : str
        Unit label for `E` ("eV", "keV", "MeV", "GeV").
    sigma_unit : str
        Unit label for `sigma` ("barn", "b", "m^2", "cm^2").
    temperature_unit : str
        Unit label for `T` ("K", "MK", "GK", "T9").
    result_unit : str
        Output unit ("m^3/mol/s" or "cm^3/mol/s").
    method : str
        Integration scheme ("trapz").
    """
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if E.shape != sigma.shape:
        raise ValueError("E and sigma must have the same shape.")
    if E.ndim != 1:
        raise ValueError("E and sigma must be one-dimensional sequences.")
    if E.size < 2:
        raise ValueError("Provide at least two sampling points for integration.")

    T_input = np.atleast_1d(np.asarray(T, dtype=float))

    if np.any(T_input <= 0.0):
        raise ValueError("Temperatures must be positive.")

    E_J = _energy_to_joules(E, unit=energy_unit)
    sigma_m2 = _sigma_to_m2(sigma, unit=sigma_unit)
    T_K = _temperature_to_kelvin(T_input, unit=temperature_unit)

    order = np.argsort(E_J)
    E_J = E_J[order]
    sigma_m2 = sigma_m2[order]

    if np.any(np.diff(E_J) <= 0.0):
        raise ValueError("Energy grid must be strictly increasing.")

    if method.lower() != "trapz":
        raise ValueError(f"Unsupported integration method '{method}'.")

    mu = reduced_mass(m1, m2)
    if mu <= 0.0:
        raise ValueError("Reduced mass must be positive.")

    # Broadcast over temperature samples.
    E_mesh = E_J[np.newaxis, :]
    sigma_mesh = sigma_m2[np.newaxis, :]
    T_mesh = T_K[:, np.newaxis]

    integrand = sigma_mesh * E_mesh * np.exp(-E_mesh / (BOLTZMANN * T_mesh))
    integral = np.trapezoid(integrand, E_J, axis=1)

    prefactor = np.sqrt(8.0 / (PI * mu)) * np.power(BOLTZMANN * T_K, -1.5)
    sigma_v = prefactor * integral
    rate = AVOGADRO * sigma_v

    rate_key = _norm_key(result_unit)
    scale = _RATE_UNIT_SCALE.get(rate_key)
    if scale is None:
        raise ValueError(f"Unsupported result unit '{result_unit}'.")
    rate_scaled = rate * scale

    return ReactionRateResult(
        temperature=T_input.copy(),
        na_sigma_v=rate_scaled,
        temperature_unit=temperature_unit,
        rate_unit=result_unit,
    )
