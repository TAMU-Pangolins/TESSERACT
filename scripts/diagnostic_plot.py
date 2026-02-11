from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nucres.physics import EV_TO_J, HBAR, MASS_PROTON, reduced_mass
from nucres.read_qvals import (
    AMU_TO_KG,
    interpret_z_token,
    mass_from_token,
    split_nuclide_token,
)
from nucres.resonance import Resonance, penetrability_P_l_mev, sigma_bw_energy_dep


@dataclass
class ReactionMeta:
    reaction: str
    z_proj: Optional[int]
    z_targ: Optional[int]
    a_proj_token: Optional[str]
    a_targ_token: Optional[str]
    s_proj: Optional[float]
    s_targ: Optional[float]
    r0_fm: float


def _parse_first_value(line: str) -> Optional[str]:
    if not line:
        return None
    part = line.split("!", 1)[0].strip()
    if not part:
        return None
    return part.split()[0]


def _parse_float_value(line: str) -> Optional[float]:
    val = _parse_first_value(line)
    if val is None:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _parse_reaction_meta(lines: list[str]) -> ReactionMeta:
    reaction = next((ln.strip() for ln in lines if ln.strip()), "Reaction")
    z_proj = None
    z_targ = None
    a_proj_token = None
    a_targ_token = None
    s_proj = None
    s_targ = None
    r0_fm = 1.25
    for line in lines[:60]:
        if "! Zproj" in line:
            z_proj = interpret_z_token(_parse_first_value(line))
        elif "! Ztarget" in line:
            z_targ = interpret_z_token(_parse_first_value(line))
        elif "! Aproj" in line:
            a_proj_token = _parse_first_value(line)
        elif "! Atarget" in line:
            a_targ_token = _parse_first_value(line)
        elif "! Jproj" in line:
            s_proj = _parse_float_value(line)
        elif "! Jtarget" in line:
            s_targ = _parse_float_value(line)
        elif "Radius parameter" in line:
            r0_val = _parse_float_value(line)
            if r0_val is not None:
                r0_fm = r0_val
    return ReactionMeta(
        reaction=reaction,
        z_proj=z_proj,
        z_targ=z_targ,
        a_proj_token=a_proj_token,
        a_targ_token=a_targ_token,
        s_proj=s_proj,
        s_targ=s_targ,
        r0_fm=r0_fm,
    )


def _mass_number_from_token(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    try:
        return int(round(float(token)))
    except ValueError:
        mass, symbol = split_nuclide_token(token)
        if mass is not None:
            return int(mass)
        if symbol:
            lower = symbol.lower()
            if lower == "p":
                return 1
            if lower == "n":
                return 1
            if lower == "a":
                return 4
        return None


def _extract_resonant_block(lines: list[str]) -> pd.DataFrame:
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("resonant contribution"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Could not find 'Resonant Contribution' section.")

    header_idx = None
    for j in range(start_idx + 1, len(lines)):
        if lines[j].strip().startswith("Ecm"):
            header_idx = j
            break
    if header_idx is None:
        raise ValueError("Could not find resonant header line starting with 'Ecm'.")

    data_start = header_idx + 1
    data_end = len(lines)
    for k in range(data_start, len(lines)):
        stripped = lines[k].strip().lower()
        if stripped.startswith("*") or stripped.startswith("upper limits"):
            data_end = k
            break

    block = "\n".join(lines[header_idx:data_end])
    return pd.read_csv(StringIO(block), sep=r"\s+")


def _wigner_limit_eV(mu_kg: float, a1: int, a2: int, r0_fm: float) -> float:
    r_m = (r0_fm * 1e-15) * ((a1 ** (1 / 3)) + (a2 ** (1 / 3)))
    wigner_j = (3.0 * HBAR**2) / (2.0 * mu_kg * r_m**2)
    return wigner_j / EV_TO_J


def _sommerfeld_eta(
    Z1: int, Z2: int, A1: int, A2: int, E_mev: np.ndarray
) -> np.ndarray:
    mu_amu = (A1 * A2) / (A1 + A2)
    return 0.157489 * (Z1 * Z2) * np.sqrt(mu_amu / np.maximum(E_mev, 1e-12))


def _get_col(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic plots for a reaction.")
    parser.add_argument(
        "--input",
        default="input/22Mg(a,p)25Al.txt",
        help="Path to RatesMC template or generated .in file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (png). Defaults to outputs/diagnostics/<reaction>_diagnostic.png",
    )
    parser.add_argument("--emin", type=float, default=None, help="Minimum E_cm (MeV).")
    parser.add_argument("--emax", type=float, default=None, help="Maximum E_cm (MeV).")
    parser.add_argument(
        "--npts", type=int, default=2000, help="Number of E points for S(E)."
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display the plot window."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    meta = _parse_reaction_meta(lines)
    df = _extract_resonant_block(lines)

    ecm_keV = _get_col(df, "Ecm")
    if ecm_keV is None:
        raise ValueError("Resonant table missing Ecm column.")
    ecm_mev = np.asarray(ecm_keV, dtype=float) * 1e-3

    J = _get_col(df, "J", "Jr")
    G1 = _get_col(df, "G1")
    G2 = _get_col(df, "G2")
    L1 = _get_col(df, "L1")
    if J is None or G1 is None or G2 is None:
        raise ValueError("Resonant table missing required columns (J, G1, G2).")
    if L1 is None:
        L1 = pd.Series(np.zeros(len(df), dtype=int))

    z1 = meta.z_proj or 0
    z2 = meta.z_targ or 0
    a1 = _mass_number_from_token(meta.a_proj_token) or 1
    a2 = _mass_number_from_token(meta.a_targ_token) or 1
    s1 = meta.s_proj if meta.s_proj is not None else 0.0
    s2 = meta.s_targ if meta.s_targ is not None else 0.0

    m1 = mass_from_token(meta.a_proj_token, z1) or (a1 * AMU_TO_KG)
    m2 = mass_from_token(meta.a_targ_token, z2) or (a2 * AMU_TO_KG)
    mu = reduced_mass(m1, m2)

    wigner_limit = _wigner_limit_eV(mu, a1, a2, meta.r0_fm)

    l_vals = np.asarray(L1, dtype=int)
    g1_vals = np.asarray(G1, dtype=float)
    g2_vals = np.asarray(G2, dtype=float)
    P_er = np.array(
        [
            penetrability_P_l_mev(l, z1, z2, a1, a2, e, meta.r0_fm)
            for l, e in zip(l_vals, ecm_mev)
        ],
        dtype=float,
    )

    theta2 = np.full_like(g1_vals, np.nan, dtype=float)
    gamma2 = np.full_like(g1_vals, np.nan, dtype=float)
    for i, (Er, g1, p_er) in enumerate(zip(ecm_mev, g1_vals, P_er)):
        if g1 <= 0.0:
            continue
        if Er < 0.0:
            theta2[i] = g1
            gamma2[i] = theta2[i] * wigner_limit
            continue
        if p_er <= 0.0:
            continue
        theta2[i] = g1 / (2.0 * p_er * wigner_limit)
        gamma2[i] = theta2[i] * wigner_limit

    emin = float(np.nanmin(ecm_mev)) if args.emin is None else args.emin
    emax = float(np.nanmax(ecm_mev)) if args.emax is None else args.emax
    e_grid = np.linspace(emin, emax, int(args.npts))

    sigma = np.zeros_like(e_grid)
    used = 0
    for Er_mev, J_i, g2, l, g2_red in zip(ecm_mev, J, g2_vals, l_vals, gamma2):
        if not np.isfinite(g2_red) or g2 <= 0.0:
            continue
        r = Resonance(
            E_r=float(Er_mev) * 1e6,
            J=float(J_i),
            s1=float(s1),
            s2=float(s2),
            m1=float(m1),
            m2=float(m2),
            Gamma_i=0.0,
            Gamma_o=float(g2),
        )
        sigma += sigma_bw_energy_dep(
            e_grid * 1e6, r, z1, z2, a1, a2, int(l), gamma2=float(g2_red), r0=meta.r0_fm
        )
        used += 1

    eta = _sommerfeld_eta(z1, z2, a1, a2, e_grid)
    s_factor = (e_grid * 1e3) * sigma * np.exp(2.0 * np.pi * eta)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(e_grid, s_factor, color="black", lw=1.2)
    axes[0].set_ylabel("S(E) [keV·b]")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    valid_theta = np.isfinite(theta2) & (theta2 > 0.0)
    sc1 = axes[1].scatter(
        ecm_mev[valid_theta],
        theta2[valid_theta],
        c=l_vals[valid_theta],
        cmap="viridis",
        s=18,
    )
    axes[1].set_ylabel(r"$\\theta^2$")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    cbar1 = fig.colorbar(sc1, ax=axes[1], label="L1")
    cbar1.ax.tick_params(labelsize=9)

    valid_p = P_er > 0.0
    sc2 = axes[2].scatter(
        ecm_mev[valid_p], P_er[valid_p], c=l_vals[valid_p], cmap="viridis", s=18
    )
    axes[2].set_xlabel("E_cm [MeV]")
    axes[2].set_ylabel("P_l(E)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    cbar2 = fig.colorbar(sc2, ax=axes[2], label="L1")
    cbar2.ax.tick_params(labelsize=9)

    fig.suptitle(meta.reaction)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = (
        Path(args.output)
        if args.output is not None
        else Path("outputs")
        / "diagnostics"
        / f"{meta.reaction.replace(' ', '')}_diagnostic.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    skipped = len(ecm_mev) - used
    print(f"Resonances used for S(E): {used} (skipped {skipped}).")
    print(f"Wrote {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
