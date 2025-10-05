from dataclasses import dataclass

@dataclass(frozen=True)
class Resonance:
    E_r     : float  # eV
    J       : float
    s1      : float
    s2      : float
    m1      : float  # kg
    m2      : float  # kg
    Gamma_i : float  # eV at E_r (set to >0 if known; else set 0 and use gamma2 in energy-dependent call)
    Gamma_o : float  # eV
