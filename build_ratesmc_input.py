from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

from nucres.generator import HFBSamplerConfig, synthesize_sigma_from_hfb
from nucres.physics import MASS_PROTON
from nucres.ratesmc_export import (
    RatesMCExportOptions,
    render_rows,
    resonant_header_line,
)
from nucres.read_qvals import interpret_z_token, mass_from_token, split_nuclide_token
from nucres.resonance import penetrability_P_l_mev


def _find_resonant_block(lines: List[str]) -> Tuple[int, int, int]:
    """
    Locate the resonant contribution section and return (header_idx, data_start_idx, data_end_idx).
    data_end_idx points to the first line after the data rows (typically the next star divider).
    """
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("resonant contribution"):
            start = i
            break
    if start is None:
        raise ValueError("Could not find 'Resonant Contribution' section in template.")

    header_idx = None
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("Ecm"):
            header_idx = j
            break
    if header_idx is None:
        raise ValueError(
            "Could not find the resonant contribution header line following the section marker."
        )

    data_start = header_idx + 1
    data_end = len(lines)
    for k in range(data_start, len(lines)):
        stripped = lines[k].strip().lower()
        if stripped.startswith("*") or stripped.startswith("upper limits"):
            data_end = k
            break

    return header_idx, data_start, data_end


def _template_has_corr_frac(lines: List[str]) -> bool:
    header_idx, _, _ = _find_resonant_block(lines)
    return "corr/frac" in lines[header_idx].lower()


def _override_random_samples(lines: List[str], n_samples: int) -> None:
    """
    Replace the line that sets the number of random samples with the provided value.
    """
    pattern = re.compile(
        r"^(\s*)[-+]?\d+(?:\.\d+)?(\s*!\s*Number of random samples.*)$", re.IGNORECASE
    )
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            lines[idx] = f"{match.group(1)}{n_samples}{match.group(2)}"
            return
    raise ValueError("Could not find the 'Number of random samples' line in template.")


def inject_rows(
    template: Path,
    out_path: Path,
    rows: List[str],
    header_line: Optional[str] = None,
    n_random_samples: Optional[int] = None,
) -> None:
    """
    Replace the resonant contribution rows in the template with the provided rows and write to out_path.
    """
    text = template.read_text(encoding="utf-8").splitlines()
    if n_random_samples is not None:
        _override_random_samples(text, n_random_samples)
    header_idx, data_start, data_end = _find_resonant_block(text)
    if header_line:
        text[header_idx] = header_line
    new_lines = text[:data_start] + rows + text[data_end:]
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _sanitize_reaction_name(name: str) -> str:
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "")
    return name.strip().replace(" ", "")


@dataclass
class TemplateMetadata:
    reaction: str
    Z: Optional[int]
    A: Optional[int]
    proj_Z: Optional[int]
    s1: Optional[float]
    s2: Optional[float]
    J: Optional[float]
    proj_Z_token: Optional[str]
    proj_A_token: Optional[str]
    targ_Z_token: Optional[str]
    targ_A_token: Optional[str]


_LIGHT_PARTICLES = {
    "n": (0, 1),
    "p": (1, 1),
    "d": (1, 2),
    "t": (1, 3),
    "a": (2, 4),
    "alpha": (2, 4),
    "g": None,
    "gamma": None,
}

_PARTICLE_EXIT_CHANNELS = {"n", "p", "d", "t", "a", "alpha"}


def _resolve_paths(
    template_arg: Optional[Path], output_arg: Optional[Path], output_dir: Optional[Path]
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    script_dir = Path(__file__).resolve().parent
    default_template = script_dir / "RatesMC.in"
    template = template_arg if template_arg is not None else default_template
    if template is None or not template.exists():
        print(f"{script_dir}")
        raise FileNotFoundError(
            f"Template RatesMC.in not provided and default ./RatesMC.in not found."
        )
    return template, output_arg, output_dir


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


def _parse_int_value(line: str) -> Optional[int]:
    val = _parse_first_value(line)
    if val is None:
        return None
    try:
        return int(float(val))
    except ValueError:
        return None


def _parse_reaction_quantities(reaction: str) -> Tuple[Optional[int], Optional[str]]:
    match = re.match(r"\s*(\d+)([A-Za-z]+)", reaction)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def _mass_number_from_token(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    try:
        return int(float(token))
    except ValueError:
        return split_nuclide_token(token)[0]


def _channel_numbers_from_token(token: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if token is None:
        return None, None
    mass_number, symbol = split_nuclide_token(token)
    if symbol is None:
        return None, None
    special = _LIGHT_PARTICLES.get(symbol.lower())
    if special is not None:
        return special
    return interpret_z_token(token), mass_number


def _parse_reaction_channels(
    reaction: str,
) -> Tuple[
    Tuple[Optional[int], Optional[int]],
    Tuple[Optional[int], Optional[int]],
    Tuple[Optional[int], Optional[int]],
]:
    match = re.match(r"\s*([^(]+)\(([^,]+),([^)]+)\)\s*([^\s]+)?", reaction)
    if not match:
        return (None, None), (None, None), (None, None)
    projectile = _channel_numbers_from_token(match.group(2).strip())
    ejectile = _channel_numbers_from_token(match.group(3).strip())
    residual = _channel_numbers_from_token(match.group(4).strip() if match.group(4) else None)
    return projectile, ejectile, residual


def _reaction_exit_symbol(reaction: str) -> Optional[str]:
    match = re.match(r"\s*([^(]+)\(([^,]+),([^)]+)\)\s*([^\s]+)?", reaction)
    if not match:
        return None
    _, symbol = split_nuclide_token(match.group(3).strip())
    return symbol.lower() if symbol is not None else None


def _convert_reduced_to_partial_widths(
    resonances,
    metadata: TemplateMetadata,
    l1_default: int,
    l2_default: int,
    r0: float = 1.25,
):
    projectile = (metadata.proj_Z, _mass_number_from_token(metadata.proj_A_token))
    target = (metadata.Z, _mass_number_from_token(metadata.targ_A_token))
    _, ejectile, residual = _parse_reaction_channels(metadata.reaction)
    convert_exit = _reaction_exit_symbol(metadata.reaction) in _PARTICLE_EXIT_CHANNELS
    converted = []
    for res in resonances:
        er_mev = max(res.E_r, 0.0) * 1e-6
        g1 = res.Gamma_i
        g2 = res.Gamma_o
        l1 = res.L1 if res.L1 is not None else l1_default
        l2 = res.L2 if res.L2 is not None else l2_default

        z1, a1 = projectile
        zt, at = target
        if None not in (z1, a1, zt, at):
            p1 = penetrability_P_l_mev(l1, z1, zt, a1, at, er_mev, r0)
            g1 = 2.0 * g1 * p1

        z2, a2 = ejectile
        zr, ar = residual
        if convert_exit and None not in (z2, a2, zr, ar):
            p2 = penetrability_P_l_mev(l2, z2, zr, a2, ar, er_mev, r0)
            g2 = 2.0 * g2 * p2

        converted.append(replace(res, Gamma_i=g1, Gamma_o=g2))
    return converted


def _parse_template_metadata(lines: List[str]) -> TemplateMetadata:
    reaction_line = next((ln.strip() for ln in lines if ln.strip()), "")
    Z_target = None
    Z_target_token = None
    Z_proj = None
    Z_proj_token = None
    A_proj_token = None
    A_target_token = None
    s1 = None
    s2 = None
    header_count = 0
    for line in lines:
        if "! Ztarget" in line:
            Z_target_token = _parse_first_value(line)
            Z_target = interpret_z_token(Z_target_token)
        elif "! Zproj" in line:
            Z_proj_token = _parse_first_value(line)
            Z_proj = interpret_z_token(Z_proj_token)
        elif "! Jproj" in line:
            s1 = _parse_float_value(line)
        elif "! Jtarget" in line:
            s2 = _parse_float_value(line)
        elif "! Aproj" in line:
            A_proj_token = _parse_first_value(line)
        elif "! Atarget" in line:
            A_target_token = _parse_first_value(line)
        header_count += 1
        if (
            header_count > 40
            and Z_target is not None
            and Z_proj is not None
            and s1 is not None
            and s2 is not None
            and A_proj_token is not None
            and A_target_token is not None
        ):
            break

    header_idx, data_start, data_end = _find_resonant_block(lines)
    J_res = None
    for idx in range(data_start, data_end):
        raw = lines[idx].strip()
        if not raw or raw.startswith("!"):
            continue
        tokens = raw.split()
        if len(tokens) >= 5:
            try:
                J_res = float(tokens[4])
                break
            except ValueError:
                continue

    A_target, _ = _parse_reaction_quantities(reaction_line.split("(")[0])

    return TemplateMetadata(
        reaction=reaction_line,
        Z=Z_target,
        A=A_target,
        proj_Z=Z_proj,
        s1=s1,
        s2=s2,
        J=J_res,
        proj_Z_token=Z_proj_token,
        proj_A_token=A_proj_token,
        targ_Z_token=Z_target_token,
        targ_A_token=A_target_token,
    )


def build_ratesmc_input(args) -> None:
    template_path, output_arg, output_dir = _resolve_paths(
        args.template, args.output, args.output_dir
    )
    lines = template_path.read_text(encoding="utf-8").splitlines()
    metadata = _parse_template_metadata(lines)
    include_corr_frac = _template_has_corr_frac(lines)
    reaction_line = metadata.reaction or "RatesMC"
    auto_output = template_path.parent / f"{_sanitize_reaction_name(reaction_line)}.in"
    if output_dir is not None:
        auto_output = (
            Path(output_dir)
            / _sanitize_reaction_name(reaction_line)
            / f"{_sanitize_reaction_name(reaction_line)}.in"
        )
        auto_output.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_arg if output_arg is not None else auto_output

    Z_val = args.Z if args.Z is not None else metadata.Z
    A_val = args.A if args.A is not None else metadata.A
    s1_val = args.s1 if args.s1 is not None else metadata.s1
    s2_val = args.s2 if args.s2 is not None else metadata.s2
    J_val = args.J if args.J is not None else metadata.J
    if J_val is None and args.sample_J and A_val is not None:
        J_val = 0.5 if (A_val % 2 == 1) else 0.0
    try:
        proj_mass_auto = mass_from_token(metadata.proj_A_token, metadata.proj_Z)
    except KeyError:
        proj_mass_auto = None
    try:
        targ_mass_auto = mass_from_token(metadata.targ_A_token, metadata.Z)
    except KeyError:
        targ_mass_auto = None
    m1_val = (
        args.m1
        if args.m1 is not None
        else (proj_mass_auto if proj_mass_auto is not None else MASS_PROTON)
    )
    m2_val = (
        args.m2
        if args.m2 is not None
        else (targ_mass_auto if targ_mass_auto is not None else MASS_PROTON)
    )

    missing = []
    if Z_val is None:
        missing.append("Z")
    if A_val is None:
        missing.append("A")
    if s1_val is None:
        missing.append("s1 (projectile spin)")
    if s2_val is None:
        missing.append("s2 (target spin)")
    if J_val is None and not args.sample_J:
        missing.append("J (resonance spin)")
    if missing:
        raise ValueError(
            "Unable to infer required parameters from template; please supply: "
            + ", ".join(missing)
        )

    cfg = HFBSamplerConfig(
        Z=Z_val,
        data_root=args.data_root,
        A=A_val,
        J=J_val,
        pi=args.pi,
        s1=s1_val,
        s2=s2_val,
        m1=m1_val,
        m2=m2_val,
        Gamma_i_mean_eV=args.Gamma_i_mean_eV,
        Gamma_o_mean_eV=args.Gamma_o_mean_eV,
        delta_E_mev=args.delta_E_mev,
        E_min_mev=args.E_min_mev,
        E_max_mev=args.E_max_mev,
        n_density_points=args.n_density_points,
        n_sigma_points=args.n_sigma_points,
        U_offset_mev=args.U_offset_mev,
        seed=args.seed,
        sample_J=args.sample_J,
        auto_l1=args.auto_l1,
    )
    try:
        generated = synthesize_sigma_from_hfb(cfg)
    except FileNotFoundError as exc:
        expected = (
            Path(args.data_root)
            if args.data_root
            else Path(__file__).resolve().parent
            / "data"
            / "densities"
            / "level-densities-hfb"
        )
        raise FileNotFoundError(
            f"Missing HFB level-density file for Z={Z_val}. "
        ) from exc

    opts = RatesMCExportOptions(
        use_strength=args.use_strength,
        default_frac_unc=args.default_frac_unc,
        l1=args.l1,
        l2=args.l2,
        l3=args.l3,
        exf_keV=args.exf_kev,
        int_flag=args.int_flag,
        include_g3=args.include_g3,
        include_corr_frac=include_corr_frac,
        corr_frac_value=0,
        j_proj=args.j_proj,
        j_targ=args.j_targ,
        precision=args.precision,
    )

    export_resonances = _convert_reduced_to_partial_widths(
        generated.resonances,
        metadata,
        l1_default=args.l1,
        l2_default=args.l2,
    )
    rows, widths = render_rows(export_resonances, opts)
    header_line = resonant_header_line(widths, opts)
    inject_rows(
        template_path,
        output_path,
        rows,
        header_line=header_line,
        n_random_samples=args.n_random_samples,
    )
    count = len(rows)
    print(f"Wrote RatesMC.in with {count} resonances to {output_path}")
    if count == 0:
        print(
            "No resonances generated; check binning widths (E-min, E-max, delta-E) and related inputs."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a RatesMC.in with resonant rows from the model output."
    )
    p.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Template RatesMC.in to start from (defaults to ./RatesMC.in next to this script).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for the filled RatesMC.in (defaults to <reaction>.in next to the template, or inside --output-dir/<reaction>/ if provided).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional base directory; if set, outputs go to <output-dir>/<reaction>/<reaction>.in.",
    )

    # Export options
    strength_group = p.add_mutually_exclusive_group()
    strength_group.add_argument(
        "--use-strength",
        dest="use_strength",
        action="store_true",
        default=False,
        help="Emit omega-gamma instead of explicit widths.",
    )
    strength_group.add_argument(
        "--use-widths",
        dest="use_strength",
        action="store_false",
        help="Emit explicit partial widths instead of omega-gamma (default).",
    )
    p.add_argument(
        "--default-frac-unc",
        type=float,
        default=0.001,
        help="Fractional uncertainty to apply to Ecm and widths (default 0.001 = 0.1%%).",
    )
    p.add_argument(
        "--l1",
        type=int,
        default=0,
        help="Entrance channel orbital angular momentum L1.",
    )
    p.add_argument(
        "--l2",
        type=int,
        default=1,
        help="Exit channel orbital angular momentum / multipolarity L2.",
    )
    p.add_argument(
        "--l3",
        type=int,
        default=0,
        help="Spectator channel orbital angular momentum L3.",
    )
    p.add_argument(
        "--exf-kev",
        type=float,
        default=0.0,
        help="Excitation energy of populated level (keV).",
    )
    p.add_argument(
        "--int-flag",
        type=int,
        choices=(0, 1),
        default=1,
        help="RatesMC integration flag (0 analytical, 1 numerical).",
    )
    p.add_argument(
        "--include-g3",
        action="store_true",
        help="If set, keep G3/DG3 columns (default zeros).",
    )
    p.add_argument(
        "--j-proj",
        type=float,
        default=None,
        help="Override projectile spin for omega-gamma (defaults to Resonance.s1).",
    )
    p.add_argument(
        "--j-targ",
        type=float,
        default=None,
        help="Override target spin for omega-gamma (defaults to Resonance.s2).",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal digits for non-Ecm columns (mantissa).",
    )
    p.add_argument(
        "--n-random-samples",
        type=int,
        default=1,
        help="Number of random samples (>5000 recommended for better statistics).",
    )

    # HFB sampler / generator options
    p.add_argument(
        "--Z",
        type=int,
        default=None,
        help="Proton number for the density grid lookup (default: parsed from template).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override path to HFB density tables.",
    )
    p.add_argument(
        "--A",
        type=int,
        default=None,
        help="Mass number slice for density grid (default: parsed from template).",
    )
    p.add_argument(
        "--J",
        type=float,
        default=None,
        help="Resonance spin used in generated spectrum (default: first J in template).",
    )
    p.add_argument("--pi", type=int, choices=(-1, 1), default=1, help="Parity (+-1).")
    p.add_argument(
        "--s1",
        type=float,
        default=None,
        help="Projectile spin (default: template Jproj).",
    )
    p.add_argument(
        "--s2",
        type=float,
        default=None,
        help="Target spin (default: template Jtarget).",
    )
    p.add_argument("--m1", type=float, default=None, help="Projectile mass (kg).")
    p.add_argument("--m2", type=float, default=None, help="Target mass (kg).")
    p.add_argument(
        "--Gamma-i-mean-eV",
        type=float,
        default=1.0,
        help="Mean entrance width for PT sampling (eV).",
    )
    p.add_argument(
        "--Gamma-o-mean-eV",
        type=float,
        default=1.0,
        help="Mean exit width for PT sampling (eV).",
    )
    p.add_argument(
        "--delta-E-mev",
        type=float,
        default=0.05,
        help="Bin width for level placements (MeV).",
    )
    p.add_argument(
        "--E-min-mev", type=float, default=0.1, help="Lower energy bound (MeV)."
    )
    p.add_argument(
        "--E-max-mev", type=float, default=2.0, help="Upper energy bound (MeV)."
    )
    p.add_argument(
        "--n-density-points",
        type=int,
        default=2001,
        help="Grid points for density interpolation.",
    )
    p.add_argument(
        "--n-sigma-points",
        type=int,
        default=4000,
        help="Resolution for sigma(E) (used only for plotting/save).",
    )
    p.add_argument(
        "--U-offset-mev",
        type=float,
        default=8.0,
        help="Excitation offset added to E_lab in density lookup.",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p.add_argument(
        "--sample-J",
        dest="sample_J",
        action="store_true",
        default=True,
        help="Sample J from HFB rho_J (uses total density for placements).",
    )
    p.add_argument(
        "--no-sample-J",
        dest="sample_J",
        action="store_false",
        help="Disable sampling J from HFB rho_J and use fixed J instead.",
    )
    p.add_argument(
        "--auto-l1",
        dest="auto_l1",
        action="store_true",
        default=True,
        help="Compute L1 from J, s1, s2, and parity (assumes intrinsic parity product +1).",
    )
    p.add_argument(
        "--no-auto-l1",
        dest="auto_l1",
        action="store_false",
        help="Disable auto L1 and use the fixed --l1 value instead.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_ratesmc_input(args)


if __name__ == "__main__":
    main()
