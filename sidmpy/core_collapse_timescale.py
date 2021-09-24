import numpy as np
import astropy.units as au

def evolution_timescale_scattering_rate(rho_s, v_rms, cross_section, rescale=1.):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average proportional to

    <sigma(v) v>

    given by Equation 4 in this paper https://arxiv.org/pdf/2102.09580.pdf
    with an additional factor of 3

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param v_rms: the velocity dispersion of the halo
    :param cross_section: an instance of the cross section model
    :return: the characteristic timescale for structural evolution in Gyr
    """

    scattering_rate_cross_section = cross_section.scattering_rate_cross_section(v_rms)
    rho_s *= au.solMass / au.kpc ** 3
    scattering_rate_cross_section *= au.cm ** 2 / au.g * au.km / au.s

    rate = 3 * rho_s * scattering_rate_cross_section
    time = 1 / rate
    time_Gyr = time.to(au.Gyr)
    return rescale * time_Gyr.value

def evolution_timescale_NFW(rho_s, rs, cross_section_amplitude):
    """
    Evaluates the timescale for the evolution of SIDM profiles given after Equation 2
    of this paper https://arxiv.org/pdf/1901.00499.pdf

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the collisionless NFW profile of the same mass
    :param momentum_averaged_cross_section: the scattering cross section in cm^2/gram weighted by v^2:
    <sigma(v) v^2>
    :return: the time in Gyr from the collapse time of a halo until when it should start to core collapse
    """

    G = 4.3e-6
    a = 4 / np.sqrt(np.pi)
    v0 = np.sqrt(4 * np.pi * G * rho_s * rs ** 2)
    const = 2.136e-19  # to year^{-1}
    t_inverse = a * const * v0 * rho_s * cross_section_amplitude
    return 1e-9 / t_inverse
