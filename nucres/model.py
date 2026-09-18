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
    r"""
    User-facing request object for HFB-based spectrum and rate generation.

    Attributes
    ----------
    Z : int
        Proton number used to select the HFB density record.
    A : int
        Mass number used for spin-grid indexing.
    J : float
        Fixed resonance spin used when the downstream sampler is not sampling `J`.
    pi : int
        Parity selector, typically `+1` or `-1`.
    s1, s2 : float
        Projectile and target spins.
    m1, m2 : float
        Projectile and target masses in kg.
    Gamma_i_mean_eV, Gamma_o_mean_eV : float
        Mean entrance and exit widths in eV.
    delta_E_mev : float
        Energy-bin width in MeV for level-count sampling.
    E_min_mev, E_max_mev : float
        Requested spectrum window in MeV.
    n_density_points, n_sigma_points : int
        Grid sizes for the density interpolation and output cross section.
    U_offset_mev : float
        Excitation-energy offset used by the default \(U(E)\) mapping.
    seed : int or None
        Random seed for reproducible sampling.
    spacing_model : {"poisson", "wigner"}
        Resonance-energy spacing model. The default preserves independent
        Poisson placement; `"wigner"` enables same-J, same-parity repulsion.
    data_root : str or Path or None
        Optional override for the HFB density-table directory.
    """

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
    spacing_model: str = "poisson"

    def to_sampler_config(self) -> HFBSamplerConfig:
        """
        Convert the request into an internal `HFBSamplerConfig`.

        Returns
        -------
        HFBSamplerConfig
            Sampler configuration carrying the same physical settings and units.
        """
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
            spacing_model=self.spacing_model,
        )

    def with_overrides(self, **overrides) -> "HFBRateRequest":
        """
        Return a copy of the request with selected fields replaced.

        Parameters
        ----------
        **overrides
            Keyword arguments matching dataclass field names.

        Returns
        -------
        HFBRateRequest
            New request object with the requested replacements applied.
        """
        return replace(self, **overrides)


def generate_spectrum(request: HFBRateRequest) -> GeneratedSpectrum:
    """
    Generate a sampled spectrum for the provided request.

    Parameters
    ----------
    request : HFBRateRequest
        High-level spectrum-generation request.

    Returns
    -------
    GeneratedSpectrum
        Sampled resonance spectrum and associated metadata.
    """
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
    """
    Generate a spectrum and return the derived reaction-rate table.

    Parameters
    ----------
    request : HFBRateRequest
        Spectrum-generation request.
    temperatures : sequence of float
        Temperature samples for the reaction-rate integration.
    temperature_unit : str, default="GK"
        Unit label for `temperatures`.
    result_unit : str, default="cm^3/mol/s"
        Output rate unit.
    energy_unit : str, default="eV"
        Energy unit passed to the rate integrator.
    sigma_unit : str, default="barn"
        Cross-section unit passed to the rate integrator.

    Returns
    -------
    ReactionRateResult
        Tabulated reaction rates for the sampled spectrum.
    """
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
    """
    Convert a `ReactionRateResult` to JSON-serializable Python objects.

    Parameters
    ----------
    result : ReactionRateResult
        Reaction-rate table to serialize.

    Returns
    -------
    dict
        Plain Python containers containing temperature and rate arrays plus units.
    """
    return {
        "temperature": np.asarray(result.temperature, dtype=float).tolist(),
        "temperature_unit": result.temperature_unit,
        "rate": np.asarray(result.na_sigma_v, dtype=float).tolist(),
        "rate_unit": result.rate_unit,
    }
