from .api import (
    Resonance,
    energy_grid,
    sigma_bw_constant,
    sigma_bw_energy_dep,
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
    sample_er_sqrt_uniform, sample_er_increasing_pdf,
    porter_thomas_factors, nonhomogeneous_poisson_placements
)
from .widths import (
    mean_particle_width_from_strength,
    mean_particle_width_from_penetrability,
    fluctuate_widths
)
