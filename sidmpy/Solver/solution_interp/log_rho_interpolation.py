from scipy.interpolate import RegularGridInterpolator
import numpy as np
from sidmpy.Solver.solution_interp.tchannel_solution_1 import *
from sidmpy.Solver.solution_interp.tchannel_solution_2 import *

cross_section_normalization_tchannel = np.arange(1, 51, 1)
redshifts_tchannel = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0,
             2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.65]
v_dependence_tchannel = np.append(np.arange(10, 55, 3), 100)
mass_values_tchannel = np.arange(6, 10.25, 0.25)
points_tchannel = (v_dependence_tchannel, cross_section_normalization_tchannel, redshifts_tchannel, mass_values_tchannel)
logrho_values_tchannel = np.stack((log_rho_vpower10, log_rho_vpower13, log_rho_vpower16,
                             log_rho_vpower19, log_rho_vpower22,
                             log_rho_vpower25, log_rho_vpower28, log_rho_vpower31,
                             log_rho_vpower34, log_rho_vpower37, log_rho_vpower40,
                             log_rho_vpower43, log_rho_vpower46, log_rho_vpower49,
                             log_rho_vpower52, log_rho_vpower100))

sigmav_values_tchannel = np.stack((sigmav_vpower10, sigmav_vpower13, sigmav_vpower16,
                             sigmav_vpower19, sigmav_vpower22, sigmav_vpower25,
                             sigmav_vpower28, sigmav_vpower31, sigmav_vpower34,
                             sigmav_vpower37, sigmav_vpower40, sigmav_vpower43,
                             sigmav_vpower46, sigmav_vpower49, sigmav_vpower52,
                             sigmav_vpower100))
interp_tchannel_logrho = RegularGridInterpolator(points_tchannel, logrho_values_tchannel)
interp_tchannel_sigmav = RegularGridInterpolator(points_tchannel, sigmav_values_tchannel)

def logrho_tchannel(log_mass, z, kwargs_cross_section, delta_c_over_c, apply_delta_c_correction=True):

    """
    This computes the central density of the halo as a function of halo mass and redshift
    :param log_mass:
    :param z:
    :param delta_concentration:
    :param kwargs_cross_section:
    :param delta_c_over_c:
    :return:
    """
    norm, v_ref = kwargs_cross_section['norm'], kwargs_cross_section['v_ref']
    if norm > cross_section_normalization_tchannel[-1]:
        raise Exception('normalization must be less than '+str(cross_section_normalization_tchannel[-1]))
    if v_ref < v_dependence_tchannel[0] or v_ref > v_dependence_tchannel[-1]:
        raise Exception('normalization must be between ' + str(v_dependence_tchannel[0]) + ' and ' + str(
            str(v_dependence_tchannel[-1])))
    if log_mass < 6 or log_mass > 10:
        raise Exception('log_mass must be between 6 and 10')

    if norm < 1:
        x = (v_ref, 1., z, log_mass)
        log10_rho0_at_one = interp_tchannel_logrho(x)
        coefs = [-0.66045156,  8.47780972]
        ratio = (coefs[0] * np.log10(norm) + coefs[1]) / (coefs[0] * np.log10(1.) + coefs[1])
        log10_rho0 = log10_rho0_at_one * ratio
    else:
        x = (v_ref, norm, z, log_mass)
        log10_rho0 = interp_tchannel_logrho(x)

    if apply_delta_c_correction:
        rho0 = 10 ** log10_rho0
        coefficients = [0.78188381, 1.50528875, 1.00770058]
        correction_term = 1 + coefficients[0] * delta_c_over_c ** 2 + coefficients[1] * delta_c_over_c
        rho0 *= correction_term
        log10_rho0 = np.log10(rho0)

    return log10_rho0

def velocity_dispersion_tchannel(log_mass, z, kwargs_cross_section, delta_c_over_c, apply_delta_c_correction=True):

    """
    This computes the velocity dispersion of the halo as a function of halo mass and redshift
    :param log_mass:
    :param z:
    :param delta_concentration:
    :param kwargs_cross_section:
    :return:
    """
    norm, v_ref = kwargs_cross_section['norm'], kwargs_cross_section['v_ref']
    if norm > cross_section_normalization_tchannel[-1]:
        raise Exception('normalization must be less than '+str(cross_section_normalization_tchannel[-1]))
    if v_ref < v_dependence_tchannel[0] or v_ref > v_dependence_tchannel[-1]:
        raise Exception('normalization must be between ' + v_dependence_tchannel[0] + ' and ' + str(
            v_dependence_tchannel[-1]))
    if log_mass < 6 or log_mass > 10:
        raise Exception('log_mass must be between 6 and 10')

    if norm < 1:
        x = (v_ref, 1., z, log_mass)
        sigma_v_at_one = interp_tchannel_sigmav(x)
        coefs = [0.24046445, 0.50059457]
        ratio = (coefs[0] * np.log10(norm) + coefs[1]) / (coefs[0] * np.log10(1.) + coefs[1])
        sigma_v = sigma_v_at_one * ratio
    else:
        x = (v_ref, norm, z, log_mass)
        sigma_v = interp_tchannel_sigmav(x)

    if apply_delta_c_correction:

        coefficients = [-0.15506128, 0.34871018, 0.99950197]
        correction_term = 1 + coefficients[0] * delta_c_over_c ** 2 + coefficients[1] * delta_c_over_c
        sigma_v *= correction_term

    return sigma_v
