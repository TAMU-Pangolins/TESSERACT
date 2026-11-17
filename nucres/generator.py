from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common.densities_retrieval import (
    J_index,
    apply_hfb_corrections,
    apply_level_density_jpi,
    hfb_sigma2,
    read_hfb_cor,
    spin_cutoff_sigma2,
    spin_grid,
)

from .hfb_adapter import load_hfb_record, resolve_density_paths
from .physics import MASS_PROTON
from .rates import ReactionRateResult, na_sigma_v_from_sigma
from .resonance import Resonance, sigma_bw_constant
from .sampling import unfolded_wigner_placements


@dataclass(frozen=True)
class HFBSamplerConfig:
    r"""
    Configuration for synthesizing resonance spectra from HFB level densities.

    Attributes
    ----------
    Z : int
        Proton number used to resolve the HFB density tables.
    data_root : str or Path or None
        Optional override for the directory containing `zXXX.tab/.cor` files.
    A : int
        Mass number used when mapping HFB spin-grid columns.
    J : float
        Fixed resonance spin used when `sample_J` is disabled or as a fallback.
    pi : int
        Compound-nucleus (resonance) parity to sample: `+1` or `-1` for a
        single parity, or `0` (default) for both parities.
    s1, s2 : float
        Projectile and target spins.
    projectile_parity, target_parity : int
        Intrinsic ground-state parities of projectile and target (`+1`/`-1`).
        A resonance J^pi can be formed only by an orbital angular momentum l
        with pi = projectile_parity * target_parity * (-1)^l and
        |J - S| <= l <= J + S for some channel spin S.
    m1, m2 : float
        Projectile and target masses in kg.
    Gamma_i_mean_eV, Gamma_o_mean_eV : float
        Porter-Thomas scale parameters in eV: mean partial widths, or mean
        reduced widths when `widths_are_reduced` is true.
    delta_E_mev : float
        Bin width in MeV for Poisson sampling of level counts.
    E_min_mev, E_max_mev : float
        Energy window in MeV over which the synthetic spectrum is generated.
    n_density_points : int
        Number of points in the cached level-density grid.
    n_sigma_points : int
        Number of points in the output cross-section grid.
    U_offset_mev : float
        Offset applied in the default mapping from resonance energy to
        excitation energy \(U(E)\).
    seed : int or None
        Random seed passed to NumPy's default generator.
    sample_J : bool
        If true, sample resonance spins from the HFB record.
    auto_l1 : bool
        If true, set `L1` to the smallest allowed entrance l for each J^pi.
    drop_forbidden : bool
        If true (default), J^pi sequences that the entrance channel cannot
        form are removed before sampling, so they contribute no resonances.
    exit_kind : {"particle", "gamma"} or None
        Exit channel type, used to choose L2 per resonance. None leaves L2
        unset (the caller's fixed default is used).
    ejectile_spin, ejectile_parity : float, int
        Spin and parity of the emitted particle (ignored for gamma exits).
    final_spin, final_parity : float or None, int or None
        J^pi of the final state populated in the residual nucleus (assumed
        to be its ground state). If unknown (None), L2 is left unset.
    drop_exit_forbidden : bool
        If true (default), resonances that cannot decay to the final state
        (no allowed l, or a gamma transition 0 -> 0) are dropped: they would
        contribute nothing to this reaction.
    spacing_model : {"poisson", "wigner"}
        Model used to place resonance energies. `"poisson"` preserves the
        independent placement used historically. `"wigner"` generates a
        Wigner-surmise renewal ladder independently for every fixed
        :math:`J^\pi` sequence in unfolded HFB-density space.
    widths_are_reduced : bool
        If true, the sampled `Gamma_i`/`Gamma_o` are reduced widths that the
        caller converts to partial widths (as `build_ratesmc_input.py` does).
        No cross section is then computed, since summing Breit-Wigner terms
        over reduced widths would be meaningless; `sigma_barns` is None.
    Gamma_i_dof, Gamma_o_dof : float
        Degrees of freedom of the mean-preserving chi-square width draws.
        Positive infinity disables fluctuations.
    use_hfb_corrections : bool
        If true, renormalise the HFB level density with the RIPL-3 `.cor`
        (ctable, ptable) entry for this isotope, when the `.cor` file has
        one; isotopes without an entry keep the raw table.
    hfb_correction : (float, float) or None
        Explicit (ctable, ptable) for the compound nucleus, used instead of
        the `.cor` entry (e.g. interpolated from neighbours for an isotope
        RIPL-3 does not list). Applied whenever it is set.
    level_density_jpi : {"hfb", "equilibrated"}
        J^pi split of the compound density. `"hfb"` (default) keeps the
        table's own rho(U, J, pi). `"equilibrated"` uses
        rho(U, J, pi) = 1/2 rho_tot(U) P(J; sigma^2(U)) with rho_tot the
        (corrected) table total and P the spin-cutoff distribution
        (`common.densities_retrieval.equilibrate_jpi`), applied after any
        correction, so it changes only the J^pi split, not the level count.
    spin_cutoff_sigma2 : float or None
        Constant sigma^2 for the equilibrated split; None uses the RIPL-3
        back-shifted Fermi-gas systematic (`spin_cutoff_sigma2`).
    cluster_density : float
        Strong-alpha ("cluster") states per MeV added on top of the
        Porter-Thomas widths; 0 (default) disables them. See
        `_apply_cluster_states`.
    cluster_window_mev : float
        Cluster states lie within this many MeV above the entrance threshold
        (intersected with [E_min_mev, E_max_mev]).
    cluster_theta2_median, cluster_sigma_ln, cluster_theta2_max : float
        Log-normal distribution of the cluster theta^2 (median and log
        width), truncated above at `cluster_theta2_max`.
    cluster_l_weights : tuple of float or None
        Relative weights of entrance l = 0, 1, 2, ... for cluster states
        (l beyond the tuple get weight 0); None weights every allowed l
        equally.
    cluster_wigner_eV : float or None
        Entrance Wigner limit hbar^2 / (mu R^2) in eV, converting theta^2 to
        the reduced widths sampled here. Required when clusters are on.
    """

    Z: int
    data_root: Optional[str | Path] = None
    A: int = 24
    J: float = 1.0
    pi: int = 0
    s1: float = 0.5
    s2: float = 0.5
    projectile_parity: int = 1
    target_parity: int = 1
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
    sample_J: bool = True
    auto_l1: bool = True
    drop_forbidden: bool = True
    exit_kind: Optional[str] = None
    ejectile_spin: float = 0.5
    ejectile_parity: int = 1
    final_spin: Optional[float] = None
    final_parity: Optional[int] = None
    drop_exit_forbidden: bool = True
    spacing_model: str = "poisson"
    widths_are_reduced: bool = False
    Gamma_i_dof: float = 1.0
    Gamma_o_dof: float = 1.0
    use_hfb_corrections: bool = False
    hfb_correction: Optional[Tuple[float, float]] = None
    level_density_jpi: str = "hfb"
    spin_cutoff_sigma2: Optional[float] = None
    cluster_density: float = 0.0
    cluster_window_mev: float = 4.0
    cluster_theta2_median: float = 0.45
    cluster_sigma_ln: float = 0.9
    cluster_theta2_max: float = 2.0
    cluster_l_weights: Optional[Tuple[float, ...]] = None
    cluster_wigner_eV: Optional[float] = None


# Salt of the cluster-state RNG stream: drawn from [seed, salt], independent of
# the ladder and width stream (seed), so runs with and without clusters share
# their levels and Porter-Thomas widths.
_CLUSTER_STREAM_SALT = 7341


@dataclass
class GeneratedSpectrum:
    """
    Sampled spectrum, resonance list, and auxiliary metadata.

    Attributes
    ----------
    energy_MeV : numpy.ndarray
        Output energy grid in MeV.
    sigma_barns : numpy.ndarray or None
        Total cross section on `energy_MeV`, in barns; None when the sampled
        widths are reduced widths (`widths_are_reduced`).
    resonances : list of Resonance
        Sampled resonance population used to build the spectrum.
    rho_energy_MeV : numpy.ndarray
        Energy grid used for the HFB density evaluation, in MeV.
    rho_levels_per_MeV : numpy.ndarray
        Level density evaluated on `rho_energy_MeV`, in levels/MeV.
    metadata : dict
        Auxiliary scalar information such as expected level count, sample size,
        widths, and masses used by downstream consumers.
    channel_rho : dict or None
        Level density (levels/MeV) on `rho_energy_MeV` of each sampled
        (J, parity) sequence, for spacings D_Jpi = 1/rho_Jpi.
    """

    energy_MeV: np.ndarray
    sigma_barns: Optional[np.ndarray]
    resonances: List[Resonance]
    rho_energy_MeV: np.ndarray
    rho_levels_per_MeV: np.ndarray
    metadata: Dict[str, Any]
    channel_rho: Optional[Dict[Tuple[float, int], np.ndarray]] = None

    @property
    def energy_eV(self) -> np.ndarray:
        """Return the sampled energy grid in eV."""
        return self.energy_MeV * 1e6

    def compute_rate(
        self,
        temperatures,
        *,
        temperature_unit: str = "GK",
        result_unit: str = "cm^3/mol/s",
        energy_unit: str = "eV",
        sigma_unit: str = "barn",
        m1: Optional[float] = None,
        m2: Optional[float] = None,
    ) -> ReactionRateResult:
        r"""
        Integrate the stored cross section over the requested temperatures.

        Parameters
        ----------
        temperatures : float or array-like
            Temperature samples to evaluate.
        temperature_unit : str, default="GK"
            Unit label understood by `na_sigma_v_from_sigma`.
        result_unit : str, default="cm^3/mol/s"
            Output rate unit.
        energy_unit : str, default="eV"
            Unit label passed to `na_sigma_v_from_sigma`. The stored grid is
            converted from MeV to eV before integration.
        sigma_unit : str, default="barn"
            Unit label for `sigma_barns`.
        m1, m2 : float or None
            Optional mass overrides in kg. When omitted, values are pulled from
            `metadata` and fall back to proton mass.

        Returns
        -------
        ReactionRateResult
            Tabulated reaction-rate result.
        """
        if self.sigma_barns is None:
            raise ValueError(
                "This spectrum was sampled with reduced widths "
                "(widths_are_reduced=True), so it has no cross section. Convert "
                "the widths to partial widths first, or sample with partial widths."
            )
        m1_eff = m1 if m1 is not None else self.metadata.get("m1", MASS_PROTON)
        m2_eff = m2 if m2 is not None else self.metadata.get("m2", MASS_PROTON)
        return na_sigma_v_from_sigma(
            self.energy_eV,
            self.sigma_barns,
            temperatures,
            m1=m1_eff,
            m2=m2_eff,
            energy_unit=energy_unit,
            sigma_unit=sigma_unit,
            temperature_unit=temperature_unit,
            result_unit=result_unit,
        )


def _interp_rhoJ_at_U(block, U_mev: float) -> np.ndarray:
    U_grid = block.U
    rhoJ = block.rho_J
    if U_mev <= U_grid[0]:
        return rhoJ[0, :]
    if U_mev >= U_grid[-1]:
        return rhoJ[-1, :]
    idx = int(np.searchsorted(U_grid, U_mev) - 1)
    x0, x1 = U_grid[idx], U_grid[idx + 1]
    y0, y1 = rhoJ[idx, :], rhoJ[idx + 1, :]
    t = (U_mev - x0) / (x1 - x0)
    return y0 * (1.0 - t) + y1 * t


def _allowed_L_values(
    J: float, s1: float, s2: float, pi_res: int, pi_intrinsic: int = 1
) -> List[int]:
    """
    Entrance orbital angular momenta l that can form a J^pi_res resonance.

    `pi_intrinsic` is the product of the projectile and target intrinsic
    parities, so the parity condition is pi_res = pi_intrinsic * (-1)^l.
    """
    min_I = abs(s1 - s2)
    max_I = s1 + s2
    start = int(round(2 * min_I))
    end = int(round(2 * max_I))
    wanted = int(pi_res) * int(pi_intrinsic)
    L_vals: List[int] = []
    for two_I in range(start, end + 1, 2):
        I = two_I / 2.0
        L_min = int(np.ceil(abs(J - I)))
        L_max = int(np.floor(J + I))
        for L in range(L_min, L_max + 1):
            if (-1) ** L == wanted:
                L_vals.append(L)
    return sorted(set(L_vals))


def _pick_L1(
    J: float, s1: float, s2: float, pi_res: int, pi_intrinsic: int = 1
) -> Optional[int]:
    allowed = _allowed_L_values(J, s1, s2, pi_res, pi_intrinsic)
    if not allowed:
        return None
    return int(allowed[0])


def _exit_L_values(
    J: float,
    pi_res: int,
    exit_kind: str,
    final_spin: float,
    final_parity: int,
    ejectile_spin: float = 0.5,
    ejectile_parity: int = 1,
) -> List[int]:
    """
    Allowed exit-channel angular momenta for a J^pi_res resonance decaying
    to a final state J_f^pi_f.

    Particle exits: orbital l with |J - S| <= l <= J + S for a channel spin
    S = |s_e - J_f| .. s_e + J_f, and pi_res = pi_e * pi_f * (-1)^l.
    Gamma exits: multipolarities L = max(1, |J - J_f|) .. J + J_f (no
    monopole, so 0 -> 0 has none); each L is E or M according to the parity
    change, so every L in that range is allowed.
    """
    if exit_kind == "gamma":
        L_min = max(1, int(round(abs(J - final_spin))))
        L_max = int(round(J + final_spin))
        return list(range(L_min, L_max + 1))
    if exit_kind == "particle":
        return _allowed_L_values(
            J, ejectile_spin, final_spin, pi_res, ejectile_parity * final_parity
        )
    raise ValueError(f"exit_kind must be 'particle' or 'gamma'; got {exit_kind!r}")


def _pick_L2(J: float, pi_res: int, config: "HFBSamplerConfig") -> Optional[int]:
    """Lowest allowed exit L (see _exit_L_values); None if none is allowed."""
    allowed = _exit_L_values(
        J, pi_res, config.exit_kind, config.final_spin, config.final_parity,
        config.ejectile_spin, config.ejectile_parity,
    )
    return int(allowed[0]) if allowed else None


def _is_natural_parity(J: float, parity: int) -> bool:
    """pi = (-1)^J for integer J; half-integer J has no natural parity, so any."""
    if abs(J - round(J)) > 1e-6:
        return True
    return int(parity) == (-1) ** int(round(J))


def _truncated_lognormal(rng, n: int, median: float, sigma_ln: float, upper: float) -> np.ndarray:
    """`n` log-normal draws (median, log width sigma_ln) truncated above at `upper`."""
    from scipy.special import ndtr, ndtri

    if sigma_ln == 0.0:
        return np.full(n, min(median, upper))
    p_max = float(ndtr(np.log(upper / median) / sigma_ln))
    return median * np.exp(sigma_ln * ndtri(rng.random(n) * p_max))


def _apply_cluster_states(
    resonances: List[Resonance],
    channels: List[tuple[float, int]],
    config: "HFBSamplerConfig",
    pi_intrinsic: int,
) -> tuple[List[Resonance], Dict[str, Any]]:
    """
    Enhance the entrance width of a few existing levels into alpha-cluster states.

    N ~ Poisson(cluster_density * width of the window [max(E_min, 0),
    min(E_max, cluster_window_mev)]) states are drawn. Each gets an energy
    uniform in the window, an entrance l from `cluster_l_weights` among the l
    that are the lowest entrance l (L1, the one RatesMC uses) of some formable
    natural-parity J^pi sequence with levels in the window, and theta^2 from
    the truncated log-normal. The not-yet-chosen level in the window of such
    a sequence nearest to that energy gets
    its reduced width raised to max(current, theta^2 * cluster_wigner_eV).
    The level count is unchanged. All draws use their own stream, so the
    ladder and the Porter-Thomas widths do not depend on the cluster settings.
    """
    if not config.widths_are_reduced:
        raise ValueError("Cluster states need reduced widths (widths_are_reduced=True).")
    if config.cluster_wigner_eV is None or not config.cluster_wigner_eV > 0.0:
        raise ValueError("Cluster states need the entrance Wigner limit cluster_wigner_eV > 0.")
    if not (config.cluster_theta2_median > 0.0 and config.cluster_theta2_max > 0.0
            and config.cluster_sigma_ln >= 0.0 and config.cluster_window_mev > 0.0):
        raise ValueError("Cluster theta^2 median, maximum and window must be positive "
                         "and sigma_ln non-negative.")

    def l1_of(J: float, parity: int) -> Optional[int]:
        return _pick_L1(J, config.s1, config.s2, parity, pi_intrinsic)

    lo = max(config.E_min_mev, 0.0)
    hi = min(config.E_max_mev, config.cluster_window_mev)
    span = max(hi - lo, 0.0)
    E_res = np.array([res.E_r * 1e-6 for res in resonances], dtype=float)
    in_window = (E_res >= lo) & (E_res <= hi)

    seq_l = {ch: l1_of(*ch) for ch in channels if _is_natural_parity(*ch)}
    # Entrance l of each level's sequence; -1 for unnatural parity, no l, or
    # outside the window. Only l with levels in the window can be drawn (the
    # spin grid runs to J ~ 50, whose sequences hold no levels here).
    res_l = np.array([
        -1 if seq_l.get((res.J, res.parity)) is None else seq_l[(res.J, res.parity)]
        for res in resonances
    ], dtype=int)
    res_l[~in_window] = -1
    l_vals = sorted({int(l) for l in res_l if l >= 0})
    if config.cluster_l_weights is None:
        weights = np.ones(len(l_vals))
    else:
        w = list(config.cluster_l_weights)
        weights = np.array([float(w[l]) if l < len(w) else 0.0 for l in l_vals])
    if np.any(weights < 0.0):
        raise ValueError("cluster_l_weights must be non-negative.")

    rng = np.random.default_rng(
        None if config.seed is None else [int(config.seed), _CLUSTER_STREAM_SALT]
    )
    n = int(rng.poisson(config.cluster_density * span)) if span > 0.0 else 0
    # Draw every random number up front, in a fixed order.
    E_draw = rng.uniform(lo, hi, n) if n else np.empty(0)
    u_l = rng.random(n)
    theta2 = _truncated_lognormal(rng, n, config.cluster_theta2_median,
                                  config.cluster_sigma_ln, config.cluster_theta2_max)

    taken = np.zeros(len(resonances), dtype=bool)
    out = list(resonances)
    states: List[Dict[str, Any]] = []
    n_unplaced = 0
    for k in range(n):
        if not l_vals or weights.sum() <= 0.0:
            n_unplaced += 1
            continue
        cdf = np.cumsum(weights) / weights.sum()
        l = int(l_vals[min(int(np.searchsorted(cdf, u_l[k], side="right")), len(l_vals) - 1)])
        cand = np.flatnonzero((res_l == l) & ~taken)
        if cand.size == 0:
            n_unplaced += 1
            continue
        i = int(cand[np.argmin(np.abs(E_res[cand] - E_draw[k]))])
        taken[i] = True
        res = out[i]
        gamma2 = float(theta2[k]) * float(config.cluster_wigner_eV)
        out[i] = replace(res, Gamma_i=max(float(res.Gamma_i), gamma2))
        states.append({
            "E_MeV": float(E_res[i]),
            "E_drawn_MeV": float(E_draw[k]),
            "J": float(res.J),
            "parity": int(res.parity),
            "l": l,
            "theta2": float(theta2[k]),
            "theta2_before": float(res.Gamma_i) / float(config.cluster_wigner_eV),
        })
    info = {"n_requested": n, "n_unplaced": n_unplaced, "allowed_l": l_vals,
            "states": states}
    return out, info


def _selected_parities(pi: int) -> tuple[int, ...]:
    pi = int(pi)
    if pi == 0:
        return (+1, -1)
    if pi in (+1, -1):
        return (pi,)
    raise ValueError(f"pi must be +1, -1, or 0 (both parities); got {pi!r}.")


def synthesize_sigma_from_hfb(config: HFBSamplerConfig) -> GeneratedSpectrum:
    r"""
    Generate a synthetic cross section by sampling resonances from HFB densities.

    The workflow is:

    1. Build the spin- and parity-resolved HFB level density \(\rho_{J\pi}(E)\)
       of the compound nucleus over the configured energy range, for every
       J^pi sequence (or only the fixed `J` when `sample_J` is false, and only
       one parity when `pi` is nonzero).
    2. Drop the J^pi sequences the entrance channel cannot form (no l satisfies
       the angular-momentum and parity selection rules), when
       `drop_forbidden` is true.
    3. Place resonances from the remaining sequences. `"poisson"` draws counts
       per `delta_E_mev` bin from the summed density and assigns each level a
       J^pi with probability proportional to \(\rho_{J\pi}(E_r)\);
       `"wigner"` builds an independent Wigner ladder per J^pi sequence.
    4. Sample widths from Porter-Thomas fluctuations and sum the resulting
       Breit-Wigner contributions.

    Parameters
    ----------
    config : HFBSamplerConfig
        Sampling configuration, units, and HFB lookup settings.

    Returns
    -------
    GeneratedSpectrum
        Synthetic spectrum, sampled resonances, level-density grid, and metadata.

    Notes
    -----
    The returned `sigma_barns` is evaluated on a uniform MeV grid, while each
    resonance energy stored in `resonances` is recorded in eV.
    Missing HFB files propagate as `FileNotFoundError` from the adapter layer.
    """
    for dof in (config.Gamma_i_dof, config.Gamma_o_dof):
        if dof != np.inf and (not np.isfinite(dof) or dof <= 0):
            raise ValueError("Width degrees of freedom must be positive or infinity")

    spacing_model = str(config.spacing_model).strip().lower()
    if spacing_model not in {"poisson", "wigner"}:
        raise ValueError(
            "spacing_model must be either 'poisson' or 'wigner'; "
            f"got {config.spacing_model!r}."
        )
    parities = _selected_parities(config.pi)
    pi_intrinsic = int(config.projectile_parity) * int(config.target_parity)

    rng = np.random.default_rng(config.seed)

    def _U_of_E_mev(E_eV: np.ndarray) -> np.ndarray:
        return np.asarray(E_eV, dtype=float) * 1e-6 + float(config.U_offset_mev)

    # Resolve the HFB level-density files once, up front, so the exact files
    # and the isotope actually tabulated in them (record.header.Z/A) can be
    # reported in the output metadata, and so the one load (with corrections
    # applied if requested) feeds every J^pi channel density built below.
    tab_p, cor_p = resolve_density_paths(config.Z, data_root=config.data_root)
    record = load_hfb_record(
        tab_path=str(tab_p),
        cor_path=(str(cor_p) if cor_p is not None else None),
        A=config.A,
        use_corrections=(config.use_hfb_corrections and config.hfb_correction is None),
        warn_if_ignored=False,
    )
    if config.hfb_correction is not None:
        key = (record.header.Z, record.header.A)
        record = apply_hfb_corrections(record, {key: tuple(map(float, config.hfb_correction))})
        hfb_corrections_applied = True
    else:
        hfb_corrections_applied = bool(
            config.use_hfb_corrections and cor_p is not None
            and (record.header.Z, record.header.A) in read_hfb_cor(str(cor_p))
        )
    sigma2_table = hfb_sigma2(record)
    record = apply_level_density_jpi(record, config.level_density_jpi, config.spin_cutoff_sigma2)

    # ── Spin/parity-resolved densities rho_{J pi}(E) on the energy grid ──────
    E_mev = np.linspace(config.E_min_mev, config.E_max_mev, config.n_density_points)
    U_values = np.asarray(_U_of_E_mev(E_mev * 1e6), dtype=float)
    J_grid = spin_grid(config.A)
    J_columns = (
        range(len(J_grid)) if config.sample_J else [J_index(config.J, config.A)]
    )

    channels: List[tuple[float, int]] = []
    channel_rho: List[np.ndarray] = []
    dropped: List[tuple[float, int]] = []
    expected_dropped = 0.0
    for parity in parities:
        block = record.positive if parity == +1 else record.negative
        if block.rho_J.shape[1] != len(J_grid):
            raise ValueError(
                "HFB spin-density columns do not match the physical spin grid: "
                f"{block.rho_J.shape[1]} columns versus {len(J_grid)} spins."
            )
        for idx in J_columns:
            J_val = float(J_grid[idx])
            rho = np.interp(
                U_values,
                block.U,
                block.rho_J[:, idx],
                left=block.rho_J[0, idx],
                right=block.rho_J[-1, idx],
            )
            rho = np.clip(rho, a_min=0.0, a_max=None)
            formable = bool(
                _allowed_L_values(J_val, config.s1, config.s2, parity, pi_intrinsic)
            )
            if config.drop_forbidden and not formable:
                dropped.append((J_val, parity))
                expected_dropped += float(np.trapezoid(rho, E_mev))
                continue
            channels.append((J_val, parity))
            channel_rho.append(rho)

    rho_matrix = (
        np.column_stack(channel_rho)
        if channel_rho
        else np.zeros((E_mev.size, 0), dtype=float)
    )
    rho_per_mev = rho_matrix.sum(axis=1)
    total_levels = float(np.trapezoid(rho_per_mev, E_mev))
    U_mid = float(_U_of_E_mev(np.array([0.5 * (config.E_min_mev + config.E_max_mev) * 1e6]))[0])
    U_lo = float(U_values[0])

    def _sigma2_at(U: float) -> Dict[str, Any]:
        table = float(np.interp(U, record.positive.U, sigma2_table))
        if config.spin_cutoff_sigma2 is not None:
            used = float(config.spin_cutoff_sigma2)
        else:
            used = float(spin_cutoff_sigma2(U, record.header.Z, record.header.A))
        return {"U_MeV": U, "systematic_or_override": used,
                "hfb_table": None if not np.isfinite(table) else table}

    n_bins = max(
        1, int(np.ceil((config.E_max_mev - config.E_min_mev) / config.delta_E_mev))
    )
    edges_mev = np.linspace(config.E_min_mev, config.E_max_mev, n_bins + 1)
    lambdas = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        a, b = edges_mev[k], edges_mev[k + 1]
        mask = (E_mev >= a) & (E_mev <= b)
        if np.count_nonzero(mask) >= 2:
            lambdas[k] = np.trapezoid(rho_per_mev[mask], E_mev[mask])
        else:
            ra = np.interp(a, E_mev, rho_per_mev)
            rb = np.interp(b, E_mev, rho_per_mev)
            lambdas[k] = 0.5 * (ra + rb) * (b - a)

    nz_bins = int((lambdas > 0).sum())

    def sample_E_in_bin(a_mev: float, b_mev: float, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=float)
        mask = (E_mev >= a_mev) & (E_mev <= b_mev)
        Ex = E_mev[mask]
        rhx = rho_per_mev[mask]
        if Ex.size == 0:
            Ex = np.array([a_mev, b_mev], dtype=float)
            rhx = np.array(
                [
                    np.interp(a_mev, E_mev, rho_per_mev),
                    np.interp(b_mev, E_mev, rho_per_mev),
                ],
                dtype=float,
            )
        else:
            if Ex[0] > a_mev:
                Ex = np.concatenate([[a_mev], Ex])
                rhx = np.concatenate([[np.interp(a_mev, E_mev, rho_per_mev)], rhx])
            if Ex[-1] < b_mev:
                Ex = np.concatenate([Ex, [b_mev]])
                rhx = np.concatenate([rhx, [np.interp(b_mev, E_mev, rho_per_mev)]])
        dE = np.diff(Ex)
        accum = np.concatenate([[0.0], np.cumsum(0.5 * (rhx[:-1] + rhx[1:]) * dE)])
        total = accum[-1]
        if total <= 0:
            return rng.uniform(a_mev, b_mev, n)
        u = rng.random(n) * total
        return np.interp(u, accum, Ex)

    # ── Place levels: list of (E_r [MeV], channel index) ─────────────────────
    levels: List[tuple[float, int]] = []
    n_ladders = 0
    if channels and spacing_model == "poisson":
        counts = rng.poisson(lambdas)
        Er_all = np.concatenate(
            [
                sample_E_in_bin(edges_mev[k], edges_mev[k + 1], int(Nk))
                for k, Nk in enumerate(counts)
            ]
        ) if counts.sum() else np.empty(0, dtype=float)
        if Er_all.size:
            # J^pi for each level, with probability proportional to rho_{J pi}(E_r).
            weights = np.column_stack(
                [np.interp(Er_all, E_mev, rho_matrix[:, c]) for c in range(len(channels))]
            )
            cdf = np.cumsum(weights, axis=1)
            u = rng.random(Er_all.size) * cdf[:, -1]
            picks = np.minimum((cdf < u[:, None]).sum(axis=1), len(channels) - 1)
            levels = [(float(e), int(c)) for e, c in zip(Er_all, picks)]
    elif channels:
        for c in range(len(channels)):
            Er_values = unfolded_wigner_placements(E_mev, rho_matrix[:, c], rng=rng)
            if Er_values.size:
                n_ladders += 1
                levels.extend((float(Er), c) for Er in Er_values)
        levels.sort(key=lambda item: item[0])
    n_drawn = len(levels)

    E_plot_MeV = np.linspace(config.E_min_mev, config.E_max_mev, config.n_sigma_points)
    E_plot_eV = E_plot_MeV * 1e6
    sigma_tot = np.zeros_like(E_plot_MeV)
    resonances: List[Resonance] = []

    def pt_width(mean_eV: float, dof: float) -> float:
        if dof == np.inf:
            return float(mean_eV)
        return float(mean_eV * rng.chisquare(df=dof) / dof)

    choose_L2 = (
        config.exit_kind is not None
        and config.final_spin is not None
        and (config.exit_kind == "gamma" or config.final_parity is not None)
    )
    n_exit_forbidden = 0
    for Er, c in levels:
        J_val, parity = channels[c]
        L1_val = (
            _pick_L1(J_val, config.s1, config.s2, parity, pi_intrinsic)
            if config.auto_l1
            else None
        )
        L2_val = _pick_L2(J_val, parity, config) if choose_L2 else None
        if choose_L2 and L2_val is None and config.drop_exit_forbidden:
            n_exit_forbidden += 1
            continue
        resonance = Resonance(
            E_r=float(Er * 1e6),
            J=J_val,
            s1=config.s1,
            s2=config.s2,
            m1=config.m1,
            m2=config.m2,
            Gamma_i=pt_width(config.Gamma_i_mean_eV, config.Gamma_i_dof),
            Gamma_o=pt_width(config.Gamma_o_mean_eV, config.Gamma_o_dof),
            L1=L1_val,
            L2=L2_val,
            parity=parity,
        )
        resonances.append(resonance)
        if not config.widths_are_reduced:
            sigma_tot += sigma_bw_constant(E_plot_eV, resonance)

    cluster_info = None
    if config.cluster_density > 0.0:
        resonances, cluster_info = _apply_cluster_states(
            resonances, channels, config, pi_intrinsic
        )

    metadata: Dict[str, Any] = {
        "expected_levels": total_levels,
        "expected_levels_dropped": expected_dropped,
        "dropped_Jpi": dropped,
        "parities": parities,
        "n_exit_forbidden": n_exit_forbidden,
        "n_bins": n_bins,
        "nonzero_lambda_bins": nz_bins,
        "n_drawn": n_drawn,
        "spacing_model": spacing_model,
        "n_spacing_ladders": n_ladders if spacing_model == "wigner" else None,
        "seed": config.seed,
        "m1": config.m1,
        "m2": config.m2,
        "Gamma_i_mean_eV": config.Gamma_i_mean_eV,
        "Gamma_o_mean_eV": config.Gamma_o_mean_eV,
        "widths_are_reduced": config.widths_are_reduced,
        "Gamma_i_dof": config.Gamma_i_dof,
        "Gamma_o_dof": config.Gamma_o_dof,
        # Implicit level-density provenance: which files were actually read,
        # which isotope they are tabulated for (per the file's own header),
        # and whether the requested compound nucleus (config.Z/A) matches it.
        "hfb_Z_requested": config.Z,
        "hfb_A_requested": config.A,
        "hfb_tab_path": str(tab_p),
        "hfb_cor_path": (str(cor_p) if cor_p is not None else None),
        "hfb_corrections_applied": hfb_corrections_applied,
        "hfb_correction_override": (list(map(float, config.hfb_correction))
                                    if config.hfb_correction is not None else None),
        "level_density_jpi": str(config.level_density_jpi).strip().lower(),
        "spin_cutoff_sigma2_override": config.spin_cutoff_sigma2,
        "spin_cutoff_sigma2": [_sigma2_at(U_lo), _sigma2_at(U_mid)],
        "hfb_header_Z": record.header.Z,
        "hfb_header_A": record.header.A,
        "hfb_Z_mismatch": record.header.Z != config.Z,
        # read_hfb_tab now raises if config.A isn't actually in the .tab file,
        # so by the time we get here this is always False in practice; kept
        # for transparency and as a guard against a future caller that
        # catches that error and proceeds anyway.
        "hfb_A_mismatch": record.header.A != config.A,
        "hfb_deformation": {
            "beta2": record.header.beta2,
            "beta3": record.header.beta3,
            "beta4": record.header.beta4,
        },
        "sample_J": config.sample_J,
        "cluster": None if cluster_info is None else {
            "density_per_MeV": config.cluster_density,
            "window_MeV": config.cluster_window_mev,
            "theta2_median": config.cluster_theta2_median,
            "sigma_ln": config.cluster_sigma_ln,
            "theta2_max": config.cluster_theta2_max,
            "l_weights": (None if config.cluster_l_weights is None
                          else list(map(float, config.cluster_l_weights))),
            "wigner_eV": config.cluster_wigner_eV,
            "allowed_l": cluster_info["allowed_l"],
        },
        "cluster_states": None if cluster_info is None else {
            "n_requested": cluster_info["n_requested"],
            "n_unplaced": cluster_info["n_unplaced"],
            "states": cluster_info["states"],
        },
    }

    return GeneratedSpectrum(
        energy_MeV=E_plot_MeV,
        sigma_barns=None if config.widths_are_reduced else sigma_tot,
        resonances=resonances,
        rho_energy_MeV=E_mev,
        rho_levels_per_MeV=rho_per_mev,
        metadata=metadata,
        channel_rho={ch: rho_matrix[:, c] for c, ch in enumerate(channels)},
    )
