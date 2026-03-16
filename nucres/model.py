from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .generator import GeneratedSpectrum, HFBSamplerConfig, synthesize_sigma_from_hfb
from .physics import MASS_PROTON
from .rates import ReactionRateResult


@dataclass(frozen=True)
class HFBRateRequest:
    """User-facing request object for HFB-based spectrum and rate generation."""

    Z: int
    A: int
    J: float
    pi: int
    s1: float = 0.5
    s2: float = 0.5
    m1: float = MASS_PROTON
    m2: float = MASS_PROTON
    Gamma_i_mean_eV: float = 1.0
    Gamma_o_mean_eV: float = 1.0
    delta_E_mev: float = 0.05
    E_min_mev: float = 0.1
    E_max_mev: float = 2.0
    n_density_points: int = 2001
    n_sigma_points: int = 4000
    U_offset_mev: float = 8.0
    seed: Optional[int] = None
    data_root: Optional[str | Path] = None

    def to_sampler_config(self) -> HFBSamplerConfig:
        """Convert the request into an internal `HFBSamplerConfig`."""
        return HFBSamplerConfig(
            Z=self.Z,
            data_root=self.data_root,
            A=self.A,
            J=self.J,
            pi=self.pi,
            s1=self.s1,
            s2=self.s2,
            m1=self.m1,
            m2=self.m2,
            Gamma_i_mean_eV=self.Gamma_i_mean_eV,
            Gamma_o_mean_eV=self.Gamma_o_mean_eV,
            delta_E_mev=self.delta_E_mev,
            E_min_mev=self.E_min_mev,
            E_max_mev=self.E_max_mev,
            n_density_points=self.n_density_points,
            n_sigma_points=self.n_sigma_points,
            U_offset_mev=self.U_offset_mev,
            seed=self.seed,
        )

    def with_overrides(self, **overrides) -> "HFBRateRequest":
        """Return a copy of the request with selected fields replaced."""
        return replace(self, **overrides)


def generate_spectrum(request: HFBRateRequest) -> GeneratedSpectrum:
    """Generate a sampled spectrum for the provided request."""
    return synthesize_sigma_from_hfb(request.to_sampler_config())


def compute_rate_table(
    request: HFBRateRequest,
    temperatures: Sequence[float],
    *,
    temperature_unit: str = "GK",
    result_unit: str = "cm^3/mol/s",
    energy_unit: str = "eV",
    sigma_unit: str = "barn",
) -> ReactionRateResult:
    """Generate a spectrum and return the derived reaction-rate table."""
    spectrum = generate_spectrum(request)
    return spectrum.compute_rate(
        temperatures,
        temperature_unit=temperature_unit,
        result_unit=result_unit,
        energy_unit=energy_unit,
        sigma_unit=sigma_unit,
        m1=request.m1,
        m2=request.m2,
    )


def rate_table_as_dict(result: ReactionRateResult) -> dict:
    """Convert a `ReactionRateResult` to JSON-serializable Python objects."""
    return {
        "temperature": np.asarray(result.temperature, dtype=float).tolist(),
        "temperature_unit": result.temperature_unit,
        "rate": np.asarray(result.na_sigma_v, dtype=float).tolist(),
        "rate_unit": result.rate_unit,
    }
