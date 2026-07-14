from .config import ensure_data_root_env

ensure_data_root_env()

from .api import (
    Resonance,
    alpha_omp_metadata,
    effective_alpha_potential_mev,
    finite_size_coulomb_mev,
    jwkb_log_transmission_mev,
    make_jwkb_log_transmission_interp,
    sigma_bw_constant,
    sigma_bw_energy_dep,
    energy_grid,
    na_sigma_v_from_sigma,
    ReactionRateResult,
    HFBSamplerConfig,
    GeneratedSpectrum,
    synthesize_sigma_from_hfb,
    HFBRateRequest,
    compute_rate_table,
    generate_spectrum,
    rate_table_as_dict,
)


from .sampling import (
    sample_er_sqrt_uniform,
    sample_er_increasing_pdf,
    porter_thomas_factors,
    nonhomogeneous_poisson_placements,
    mean_particle_width_from_strength,
    mean_particle_width_from_penetrability,
    fluctuate_widths,
)
