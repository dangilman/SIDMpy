import numpy as np
import astropy.units as au
from sidmpy.Solver.util import nfw_velocity_dispersion_fromfit
from scipy.integrate import quad
from pyHalo.Halos.lens_cosmo import LensCosmo

def fraction_collapsed_halos_pool(args):
    (m1, m2, cross_section, redshift, timescale_factor) = args
    return fraction_collapsed_halos(m1, m2, cross_section, redshift, timescale_factor)

def fraction_collapsed_halos(m1, m2, cross_section, redshift, timescale_factor,
                             t_min_scale=0.5, t_max_scale=2.0, alpha=-1.9,
                             timescale_pdf='LINEAR', collapse_redshift=10., approx=True):
    """
    Returns the fraction of core-collapsed objects with mass M m1 < M < m2
    :param m1: minimum mass
    :param m2: maximum mass
    :param redshift: the redshift at which to evaluate core collapse
    :param timescale_factor: the factor multiplying the relaxation time to get a timescale for core collapse
    :param cross_section: a cross section class
    :param t_min_scale:
    :param t_max_scale:
    :param alpha: logarithmic slope of the mass function
    :param timescale_pdf:
    :return:
    """

    l = LensCosmo()
    time = l.cosmo.halo_age(redshift, collapse_redshift)
    integrand_numerator = lambda m: m ** alpha * collapse_probability_fromM(time, m, redshift, cross_section,
                                                                                 timescale_factor, t_min_scale, t_max_scale,
                                                                            timescale_pdf)
    integral_demon = (m2**(1+alpha)-m1**(1+alpha))/(1+alpha)

    if approx or abs(np.log10(m2) - np.log10(m1)) < 0.2:
        dm = m2 - m1
        numerator = integrand_numerator(0.5 * (m1 + m2)) * dm
    else:
        numerator = quad(integrand_numerator, m1, m2)[0]

    return numerator/integral_demon

def collapse_probability_fromM(t, m, halo_redshift, cross_section, timescale_factor, t_min_scale=0.5, t_max_scale=2.0,
                               type='LINEAR'):

    t_r = evolution_timescale_scattering_rate_fromM(m, halo_redshift, cross_section)
    t_collapse = timescale_factor * t_r
    if type == 'HYPERBOLIC':
        return collapse_probability_hyperbolic(t, t_collapse, t_min_scale, t_max_scale)
    elif type == 'LINEAR':
        return collapse_probability_linear(t, t_collapse, t_min_scale, t_max_scale)
    else:
        raise Exception('type must be LINEAR or HYPERBOLIC')

def collapse_probability_hyperbolic(t, t_c, t_min_scale=0.5, t_max_scale=2.0):

    t_min_scale /= 4
    t_max_scale *= 4

    if t < t_min_scale * t_c:
        return 0.0
    elif t > t_max_scale * t_c:
        return 1.0

    t_width = 0.25 * t_c
    arg = (t - t_c) / t_width
    return (1 + np.tanh(arg))/2

def collapse_probability_linear(t, t_c, t_min_scale=0.5, t_max_scale=2.0):

    if t < t_min_scale * t_c:
        return 0.0
    elif t > t_max_scale * t_c:
        return 1.0

    t_1 = t_min_scale * t_c
    t_2 = t_max_scale * t_c
    return (t - t_1) / (t_2 - t_1)

def evolution_timescale_scattering_rate_fromM(halo_mass, halo_redshift, cross_section, rescale=1.):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average proportional to

    <sigma(v) v>

    given by Equation 4 in this paper https://arxiv.org/pdf/2102.09580.pdf
    with an additional factor of 3

    :param halo mass: the mass of the halo
    :param halo_redshift: the redshift of the halo
    :param cross_section: an instance of the cross section model
    :return: the characteristic timescale for structural evolution in Gyr
    """

    try:
        from pyHalo.Halos.lens_cosmo import LensCosmo
        l = LensCosmo()
    except:
        raise Exception('could not import module pyHalo (required for this function')

    v_rms = nfw_velocity_dispersion_fromfit(halo_mass)
    c = l.NFW_concentration(halo_mass, halo_redshift, scatter=False)
    rho_s, _, _ = l.NFW_params_physical(halo_mass, c, halo_redshift)
    scattering_rate_cross_section = cross_section.scattering_rate_cross_section(v_rms)
    rho_s *= au.solMass / au.kpc ** 3
    scattering_rate_cross_section *= au.cm ** 2 / au.g * au.km / au.s

    rate = rho_s * scattering_rate_cross_section
    time = 1 / rate
    time_Gyr = time.to(au.Gyr)
    return rescale * time_Gyr.value

def evolution_timescale_scattering_rate(rho_s, v_rms, cross_section, rescale=1.):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average proportional to

    <sigma(v) v>

    given by Equation 4 in this paper https://arxiv.org/pdf/2102.09580.pdf

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
