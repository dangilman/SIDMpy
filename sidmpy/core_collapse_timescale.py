import numpy as np
import astropy.units as au

def mean_free_path(rhos, vrms, cross_section):

    """
    Calculate the mean free path given the central density, velocity dispersion, and cross_section model

    velocity-dep cross sections will be averaged over the velocity assuming MB distribution
    :return:
    """

    rhos *= au.solMass / au.kpc**3
    sigma_v = cross_section.velocity_weighted_average(vrms, 0)
    mean_v = cross_section.velocity_moment(vrms, 0)

    sigma_eff = sigma_v / mean_v * au.cm ** 2 / au.g
    mfp = 1 / (sigma_eff * rhos * np.sqrt(2))
    return mfp.to(au.kpc).value

def gravitational_scale_height(rhos, vrms):

    """
    Computes the gravitational scale height (gsh) of a halo, useful to compare with the mean free path.

    If mfp >> gsh then there are self-similar solutions for the halo evolution (https://arxiv.org/pdf/astro-ph/0110561.pdf)
    :param rhos:
    :param vrms:
    :return:
    """
    G = 4.3e-6 * au.kpc / au.solMass * au.km ** 2 / au.s ** 2
    vrms *= au.km/au.s
    rhos *= au.solMass / au.kpc ** 3
    l = np.sqrt(vrms ** 2 / (4 * np.pi * G * rhos))
    return l.to(au.kpc).value

def evolution_timescale_momentum_transfer(rhos, rs, v_rms, cross_section, rescale=1.):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the momentum-averaged
    cross section discussed <v^2 sigma(v)>/<v^2>

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the collisionless NFW profile of the same mass
    :param cross_section: the scattering cross section model
    :return: the time in Gyr from the collapse time of a halo until when it should start to core collapse
    """
    a = 4/np.sqrt(np.pi)
    G = 4.3e-6 * au.kpc / au.solMass * au.km ** 2 / au.s ** 2
    rhos *= au.solMass / au.kpc ** 3
    rs *= au.kpc
    v_scale = np.sqrt(4 * np.pi * G * rhos * rs ** 2)

    momentum_averaged_cross_section = cross_section.momentum_transfer_cross_section(v_rms)
    momentum_averaged_cross_section *= au.cm ** 2 / au.g

    rate = a * v_scale * rhos * momentum_averaged_cross_section
    time = 1/rate
    time_Gyr = time.to(au.Gyr)
    return rescale * time_Gyr.value

def evolution_timescale_scattering_rate(rho_s, v_rms, cross_section, rescale=1.):
    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average instead of the momentum average, i.e. <sigma v>/<v> instead of <sigma v^2>/<v^2>
    cross section discussed on page 20 of https://arxiv.org/pdf/2011.04679.pdf

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the collisionless NFW profile of the same mass
    :param cross_section: the cross section model
    :return: the time in Gyr from the collapse time of a halo until when it should start to core collapse
    """
    scattering_rate_cross_section = cross_section.scattering_rate_cross_section(v_rms)
    rho_s *= au.solMass / au.kpc ** 3
    scattering_rate_cross_section *= au.cm ** 2 / au.g * au.km / au.s

    rate = 3 * rho_s * scattering_rate_cross_section
    time = 1 / rate
    time_Gyr = time.to(au.Gyr)
    return rescale * time_Gyr.value

def evolution_timescale_effective_cross_section(rhos, rs, v_rms, cross_section):
    """
    Evaluates the timescale for the evolution of SIDM profiles using the an effective cross
    section <sigma(v)> i.e. <sigma(v) v^n> with n=0

    :param rhos: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the collisionless NFW profile of the same mass
    :param cross_section: the scattering cross section model
    :return: the time in Gyr from the collapse time of a halo until when it should start to core collapse
    """

    effective_cross_section = cross_section.velocity_weighted_average(v_rms, 0)

    effective_cross_section *= au.cm ** 2 / au.g

    G = 4.3e-6 * au.kpc / au.solMass * au.km ** 2 / au.s ** 2
    rhos *= au.solMass / au.kpc ** 3
    rs *= au.kpc
    v_scale = np.sqrt(4 * np.pi * G * rhos * rs ** 2)

    rate = rhos * effective_cross_section * v_scale
    time = 1 / rate
    time_Gyr = time.to(au.Gyr)
    return time_Gyr.value

def evolution_timescale_NFW(rho_s, rs, cross_section_amplitude):
    """
    Evaluates the timescale for the evolution of SIDM profiles using the momentum-averaged
    cross section discussed on page 20 of https://arxiv.org/pdf/2011.04679.pdf

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
