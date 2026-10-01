"""Non-relativistic lab <-> centre-of-mass energy conversion using AME2020 masses."""

from .read_qvals import atomic_mass_u, element_symbol_to_Z


def _mass_u(A: int, sym: str) -> float:
    Z = element_symbol_to_Z(sym)
    if Z is None:
        raise ValueError(f"Unknown element symbol: {sym!r}")
    return atomic_mass_u(Z, int(A))


def cm_over_lab(A_tar, sym_tar, A_pro, sym_pro) -> float:
    r"""
    Return \(E_{cm}/E_{lab} = m_t / (m_t + m_p)\) for a projectile on a target at rest.

    Raises
    ------
    KeyError
        If either nuclide is missing from the AME table.
    """
    m_t = _mass_u(A_tar, sym_tar)
    m_p = _mass_u(A_pro, sym_pro)
    return m_t / (m_t + m_p)


def lab_to_cm(E_lab, A_tar, sym_tar, A_pro, sym_pro):
    """
    Convert a projectile lab-frame energy to center-of-mass energy in MeV.

    Parameters
    ----------
    E_lab : float or array-like
        Projectile energy in the laboratory frame, in MeV.
    A_tar, A_pro : int
        Target and projectile mass numbers.
    sym_tar, sym_pro : str
        Target and projectile element symbols.

    Returns
    -------
    float or numpy.ndarray
        Center-of-mass energy in MeV.
    """
    return E_lab * cm_over_lab(A_tar, sym_tar, A_pro, sym_pro)


def cm_to_lab(E_cm, A_tar, sym_tar, A_pro, sym_pro):
    """
    Convert a center-of-mass energy to the projectile lab-frame energy in MeV.

    Inverse of `lab_to_cm`; same parameters.
    """
    return E_cm / cm_over_lab(A_tar, sym_tar, A_pro, sym_pro)
