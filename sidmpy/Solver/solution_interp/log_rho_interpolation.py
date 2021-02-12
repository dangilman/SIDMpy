from scipy.interpolate import RegularGridInterpolator
import numpy as np

cross_section_normalization = np.array([0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
redshifts = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0,
             2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.65]
mass_values = [6, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.]
v_dependence_powerlaw = [0., 0.2, 0.4, 0.6, 0.8]

points_tchannel = (cross_section_normalization, redshifts, mass_values)
#interp_tchannel = RegularGridInterpolator(points_tchannel, log_rho_w10)

points_power_law = (v_dependence_powerlaw, cross_section_normalization, redshifts, mass_values)
#values_power_law = np.stack((log_rho_vpower0, log_rho_vpower025, log_rho_vpower05, log_rho_vpower075))
#interp_powerlaw = RegularGridInterpolator(points_power_law, values_power_law)

def logrho_tchannel(log_mass, z, delta_concentration, kwargs_cross_section, concentration_scatter_scale=0.85):

    norm, v_ref = kwargs_cross_section['norm'], kwargs_cross_section['v_ref']
    if norm < cross_section_normalization[0]:
        norm = cross_section_normalization[0]
    elif norm > cross_section_normalization[-1]:
        norm = cross_section_normalization[-1]

    if log_mass < 6 or log_mass > 10:
        raise Exception('log_mass must be between 6 and 10')
    if v_ref != 30:
        raise Exception('TCHANNEL solution only computed for a reference velocity w = 10 km/sec')

    x = (norm, z, log_mass)
    rho0 = interp_tchannel(x)

    rho0 = 10 ** (np.log10(rho0) + delta_concentration * concentration_scatter_scale)

    return rho0

def logrho_power_law(log_mass, z, delta_concentration, kwargs_cross_section, concentration_scatter_scale=0.85):
    """

    Returns the central density of an SIDM halo with an interaction cross section parameterized as:

    sigma(v) = sigma_0 * (30 / v) ^ v_dep

    :param log_mass: log(halo mass), mass definition m_200 (no little h)
    :param z: halo redshift
    :param c0: self interacction cross section at 30 km/sec
    :param v_dep: velocity dependence of cross section
    :param delta_concentration: fractional deviation from the median halo concentration with respect to
    the m-c relation of Diemer and Joyce (2019)
    :param concentration_scatter_scale: tunes the relationship between the scatter in the m-c relation and the
    scatter on the central density. 0.3 (dex) gives a good approximation

    :return: log(rho), where rho is the central density of the SIDM halo
    """

    norm, v_dep = kwargs_cross_section['norm'], kwargs_cross_section['v_dep']
    v_ref = kwargs_cross_section['v_ref']
    if v_dep < 0:
        raise Exception('velocity dep must be >= 0.')
    elif v_dep > v_dependence_powerlaw[-1]:
        raise Exception('velocity dep must be <= 0.75')
    if v_ref != 30:
        raise Exception('POWER_LAW solution only computed for a reference velocity w = 30 km/sec')

    if norm < cross_section_normalization[0]:
        norm = cross_section_normalization[0]
    elif norm > cross_section_normalization[-1]:
        norm = cross_section_normalization[-1]

    if log_mass < 6 or log_mass > 10:
        raise Exception('log_mass must be between 6 and 10')

    x = (v_dep, norm, z, log_mass)
    rho0 = interp_powerlaw(x)
    rho0 = 10 ** (np.log10(rho0) + delta_concentration * concentration_scatter_scale)

    return rho0
