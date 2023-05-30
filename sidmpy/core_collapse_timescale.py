import numpy as np
import astropy.units as au
from sidmpy.Solver.util import nfw_velocity_dispersion_fromfit, nfw_vmax
from scipy.integrate import quad
from pyHalo.Halos.lens_cosmo import LensCosmo


def fraction_collapsed_halos_pool(args):
    (m1, m2, cross_section, redshift, timescale_factor, collapse_window) = args
    return fraction_collapsed_halos(m1, m2, cross_section, redshift, timescale_factor, collapse_window)

def collapse_prob_linear_window(rhos, rs, z, cross_section, lens_cosmo, tscale_factor, window, kwargs_disp,
                                velocity_averaging=5):

    if kwargs_disp is not None:
        v = nfw_vmax(rhos, rs)
        dissipation_factor = dissipation_timescale_impact(v, cross_section, **kwargs_disp)
        tscale_factor *= dissipation_factor

    halo_age = lens_cosmo.cosmo.halo_age(z)
    if velocity_averaging == 5:
        t_scale = evolution_timescale_v5(rhos, rs, None, cross_section)
    elif velocity_averaging == 3:
        t_scale = evolution_timescale_outmezguine(rhos, rs, None, cross_section)
    else:
        raise Exception('only velocity averaging of 3 and 5 implemented')
    collapse_timescale = tscale_factor * t_scale

    t_min = collapse_timescale - window / 2
    t_max = collapse_timescale + window / 2

    if halo_age < t_min:
        return 0.0
    elif halo_age > t_max:
        return 1.0
    else:
        return (halo_age - t_min) / (t_max - t_min)

def collapse_prob_linear_window_fromM(m, z, cross_section, lens_cosmo, tscale_factor, window, kwargs_disp,
                                velocity_averaging=5):

    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')
    c = lens_cosmo.NFW_concentration(m, z, scatter=False)
    rhos, rs, _ = lens_cosmo.NFW_params_physical(m, c, z)

    return collapse_prob_linear_window(rhos, rs, z, cross_section, lens_cosmo, tscale_factor, window, kwargs_disp,
                                velocity_averaging)

def collapse_prob_sigmoid_fromM(m, z, cross_section, lens_cosmo, tscale_factor, window, kwargs_disp,
                          velocity_averaging=5):

    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')
    c = lens_cosmo.NFW_concentration(m, z, scatter=False)
    rhos, rs, _ = lens_cosmo.NFW_params_physical(m, c, z)

    return collapse_prob_sigmoid(rhos, rs, z, cross_section, lens_cosmo, tscale_factor,
           window, kwargs_disp, velocity_averaging)

def collapse_prob_sigmoid(rhos, rs, z, cross_section, lens_cosmo, tscale_factor, window, kwargs_disp,
                          velocity_averaging=5):

    if kwargs_disp is not None:
        v_rms = nfw_velocity_dispersion_fromfit(m)
        dissipation_factor = dissipation_timescale_impact(v_rms, cross_section, **kwargs_disp)
        tscale_factor *= dissipation_factor

    halo_age = lens_cosmo.cosmo.halo_age(z)
    if velocity_averaging == 5:
        t_scale = evolution_timescale_v5(rhos, rs, None, cross_section)
    elif velocity_averaging == 3:
        t_scale = evolution_timescale_outmezguine(rhos, rs, None, cross_section)
    else:
        raise Exception('only velocity averaging of 3 and 5 implemented')
    collapse_timescale = tscale_factor * t_scale

    X = (halo_age - collapse_timescale)/(2*window)
    p = 1/(1 + np.exp(-X))

    return p


def fraction_collapsed_halos(m1, m2, cross_section, z, tscale_factor, collapse_window, lens_cosmo=None, approx=True,
                             collapse_prob_window='SIGMOID', kwargs_disp=None, velocity_averaging=5):
    assert m2 > m1

    if collapse_prob_window == 'LINEAR_WINDOW':
        collapse_prob = collapse_prob_linear_window_fromM
    elif collapse_prob_window == 'SIGMOID':
        collapse_prob = collapse_prob_sigmoid_fromM
    else:
        raise Exception('collapse_prob_window must be either LINEAR or LINEAR_WINDOW')

    if lens_cosmo is None:
        lens_cosmo = LensCosmo()

    if m2 / m1 < 2 and approx:
        return collapse_prob((m1 + m2) / 2, z, cross_section, lens_cosmo, tscale_factor, collapse_window, kwargs_disp,
                             velocity_averaging)

    def _integrand_denom(m):
        return m ** -1.9

    def _integrand_numerator(m):
        return _integrand_denom(m) * collapse_prob(m, z, cross_section, lens_cosmo, tscale_factor, collapse_window,
                                                   kwargs_disp, velocity_averaging)

    return quad(_integrand_numerator, m1, m2)[0] / quad(_integrand_denom, m1, m2)[0]

def evolution_timescale_Essig_fromM(halo_mass, halo_redshift, cross_section_constant_amplitude, lens_cosmo):

    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')

    v_rms = nfw_velocity_dispersion_fromfit(halo_mass)
    c = lens_cosmo.NFW_concentration(halo_mass, halo_redshift, scatter=False)
    rho_s, _, _ = lens_cosmo.NFW_params_physical(halo_mass, c, halo_redshift)

    c = lens_cosmo.NFW_concentration(halo_mass, halo_redshift, scatter=False)
    rho_s, rs, _ = lens_cosmo.NFW_params_physical(halo_mass, c, halo_redshift)
    rho_s *= au.solMass / au.kpc**3
    rs *= au.kpc
    rho_s_rs = rho_s * rs

    cross_section_constant_amplitude *= au.cm**2/au.g
    cross_section_constant_amplitude = cross_section_constant_amplitude.to(au.kpc**2/au.solMass)
    term1 = rho_s_rs * cross_section_constant_amplitude

    G = 4.3e-6 * au.kpc/au.solMass * (au.km/au.s)**2
    term2 = np.sqrt(4 * np.pi * G * rho_s).to(au.s**-1)
    denom = term1 * term2

    beta = 0.45
    tscale = 150/beta/denom
    time_Gyr = tscale.to(au.Gyr)

    return time_Gyr.value

def evolution_timescale_scattering_rate_fromM(halo_mass, halo_redshift, cross_section, lens_cosmo=None):

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
    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')

    v_rms = nfw_velocity_dispersion_fromfit(halo_mass)
    c = lens_cosmo.NFW_concentration(halo_mass, halo_redshift, scatter=False)
    rho_s, _, _ = lens_cosmo.NFW_params_physical(halo_mass, c, halo_redshift)
    return evolution_timescale_scattering_rate(rho_s, None, v_rms, cross_section)

def evolution_timescale_scattering_rate(rho_s, rs, v_rms, cross_section, rescale=1.):

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

    rate = rho_s * scattering_rate_cross_section
    time = 1 / rate
    time_Gyr = time.to(au.Gyr)
    return rescale * time_Gyr.value

def evolution_timescale_outmezguine_fromM(m, z, cross_section, lens_cosmo=None):

    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')
    c = lens_cosmo.NFW_concentration(m, z, scatter=False)
    rhos, rs, _ = lens_cosmo.NFW_params_physical(m, c, z)
    return evolution_timescale_outmezguine(rhos, rs, None, cross_section)

def evolution_timescale_v5_fromM(m, z, cross_section, lens_cosmo=None):

    if lens_cosmo is None:
        try:
            from pyHalo.Halos.lens_cosmo import LensCosmo
            lens_cosmo = LensCosmo()
        except:
            raise Exception('could not import module pyHalo (required for this function')
    c = lens_cosmo.NFW_concentration(m, z, scatter=False)
    rhos, rs, _ = lens_cosmo.NFW_params_physical(m, c, z)
    return evolution_timescale_v5(rhos, rs, None, cross_section)

def evolution_timescale_outmezguine(rho_s, rs, v_rms, cross_section):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average proportional to

    <sigma(v) v^3>

    given by Equation 4 in this paper https://arxiv.org/pdf/2102.09580.pdf

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the host halo
    :param v_rms: the velocity dispersion of the halo
    :param cross_section: an instance of the cross section model
    :return: the characteristic timescale for structural evolution in Gyr
    """
    G = 4.3e-6
    vmax = 1.65 * np.sqrt(G*rho_s * rs ** 2)
    energy_transfer_cross_section = cross_section.energy_transfer_cross_section(0.64 * vmax)
    sigma_0 = energy_transfer_cross_section
    t_c = (1/sigma_0) * (100/vmax) * (10**7/rho_s)
    return t_c

def evolution_timescale_v5(rho_s, rs, v_rms, cross_section):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the scattering rate
    average proportional to

    <sigma(v) v^5>

    given by Equation 4 in this paper https://arxiv.org/pdf/2102.09580.pdf

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the host halo
    :param v_rms: the velocity dispersion of the halo
    :param cross_section: an instance of the cross section model
    :return: the characteristic timescale for structural evolution in Gyr
    """
    G = 4.3e-6
    vmax = 1.65 * np.sqrt(G*rho_s * rs ** 2)
    thermally_averaged_cross_section = cross_section.v5_transfer_cross_section(vmax)
    t_c = (1/thermally_averaged_cross_section) * (100/vmax) * (10**7/rho_s)
    return t_c

def sigma_rhos_rs(cross, halo_mass, redshift):

    try:
        from pyHalo.Halos.lens_cosmo import LensCosmo
        l = LensCosmo()
    except:
        raise Exception('could not import module pyHalo (required for this function')

    c = l.NFW_concentration(halo_mass, redshift, scatter=False)
    rho_s, rs, _ = l.NFW_params_physical(halo_mass, c, redshift)
    rho_s *= au.solMass / au.kpc**3
    rs *= au.kpc
    cross *= au.cm**2 / au.g
    cross = cross.to(au.kpc**2 / au.solMass)
    return (cross * rho_s * rs).value

def evolution_timescale_NFW(rho_s, rs, v_rms, cross_section_amplitude):
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
