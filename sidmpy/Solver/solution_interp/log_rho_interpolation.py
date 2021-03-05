from scipy.interpolate import RegularGridInterpolator
import numpy as np
from sidmpy.Solver.solution_interp.tchannel_solution_table import *
# from sidmpy.Solver.solution_interp.power_law_solution_table import log_rho_vpower0, log_rho_vpower02, log_rho_vpower04, \
#     log_rho_vpower06, log_rho_vpower08

cross_section_normalization_tchannel = np.arange(1, 51, 1)

redshifts_tchannel = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0,
             2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.65]
v_dependence_tchannel = np.append(np.arange(10, 55, 3), 100)

mass_values_tchannel = np.arange(6, 10.25, 0.25)

v_dependence_powerlaw = [0., 0.2, 0.4, 0.6, 0.8]
cross_section_normalization_powerlaw = np.array([0.5, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
redshifts_powerlaw = [0, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.0,
             2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.65]
mass_values_powerlaw = [6, 6.4, 6.8, 7.2, 7.6, 8.0, 8.4, 8.8, 9.2, 9.6, 10.]

points_tchannel = (v_dependence_tchannel, cross_section_normalization_tchannel, redshifts_tchannel, mass_values_tchannel)
# values_tchannel = np.stack((log_rho_w10, log_rho_w12, log_rho_w14, log_rho_w16, log_rho_w18, log_rho_w20,
#                             log_rho_w22, log_rho_w24, log_rho_w26, log_rho_w28, log_rho_w30, log_rho_w32,
#                             log_rho_w34, log_rho_w36, log_rho_w38, log_rho_w40, log_rho_w42, log_rho_w44,
#                             log_rho_w46, log_rho_w48, log_rho_w50, log_rho_w100))
#
# interp_tchannel = RegularGridInterpolator(points_tchannel, values_tchannel)

#points_power_law = (v_dependence_powerlaw, cross_section_normalization_powerlaw, redshifts_powerlaw, mass_values_powerlaw)
#values_power_law = np.stack((log_rho_vpower0, log_rho_vpower02, log_rho_vpower04, log_rho_vpower06,
#                             log_rho_vpower08))
#interp_powerlaw = RegularGridInterpolator(points_power_law, values_power_law)

def logrho_tchannel(log_mass, z, kwargs_cross_section, delta_c_over_c):

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
    if norm < cross_section_normalization_tchannel[0] or norm > cross_section_normalization_tchannel[-1]:
        raise Exception('normalization must be between '+str(cross_section_normalization_tchannel[0])+' and '+str(cross_section_normalization_tchannel[-1]))
    if v_ref < v_dependence_tchannel[0] or v_ref > v_dependence_tchannel[-1]:
        raise Exception('normalization must be between ' + v_dependence_tchannel[0] + ' and ' + str(
            v_dependence_tchannel[-1]))
    if log_mass < 6 or log_mass > 10:
        raise Exception('log_mass must be between 6 and 10')

    x = (v_ref, norm, z, log_mass)
    log10_rho0 = interp_tchannel(x)
    rho0 = 10 ** log10_rho0
    coefficients = [0.70992004, 1.28768607, 0.984719]
    correction_term = 1 + coefficients[0] * delta_c_over_c ** 2 + coefficients[1] * delta_c_over_c
    rho0 *= correction_term

    return np.log10(rho0)
