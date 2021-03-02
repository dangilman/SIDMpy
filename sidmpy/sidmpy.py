from sidmpy.CrossSections.power_law import PowerLaw
from sidmpy.CrossSections.velocity_independent import VelocityIndependentCrossSection
from sidmpy.CrossSections.tchannel import TChannel
from sidmpy.Solver.solver import solve_profile
from sidmpy.Solver.solution_interp.log_rho_interpolation import logrho_tchannel
import numpy as np

def solve_with_interpolation(halo_mass, halo_redshift, delta_c_over_c, cross_section_type, kwargs_cross_section):

    if cross_section_type == 'POWER_LAW':
        raise Exception('net yet implemented')

    elif cross_section_type == 'TCHANNEL':
        return 10 ** logrho_tchannel(np.log10(halo_mass), halo_redshift, kwargs_cross_section, delta_c_over_c)
    else:
        raise Exception('cross section type not recognized')

def solve_from_NFW_params(rhos, rs, halo_age, cross_section_type, kwargs_cross_section, **kwargs_solver):

    """
    This routine solves for the SIDM central density given the normalization of an NFW halo rhos, the scale radius rs,
    and a specific interaction cross section
    """

    if cross_section_type == 'POWER_LAW':
        cross_section_class = PowerLaw(**kwargs_cross_section)
    elif cross_section_type == 'VELOCITY_INDEPENDENT':
        cross_section_class = VelocityIndependentCrossSection(**kwargs_cross_section)
    elif cross_section_type == 'TCHANNEL':
        cross_section_class = TChannel(**kwargs_cross_section)
    else:
        raise Exception('cross section type not recognized')

    rho0, vdis, r1 = solve_profile(rhos, rs, cross_section_class, halo_age, **kwargs_solver)
    return rho0, vdis, r1

def solve_from_Mz(M, z, cross_section_type, kwargs_cross_section, z_collapse=10., include_c_scatter=False,
                  c_scatter_add_dex=0., **kwargs_solver):
    """
    This routine solves for the SIDM central density given the normalization of an NFW halo rhos, the scale radius rs,
    and a specific interaction cross section
    """
    try:
        from pyHalo.Halos.lens_cosmo import LensCosmo
        from pyHalo.Cosmology.cosmology import Cosmology
        cosmo = Cosmology()
        lens_cosmo = LensCosmo()
    except:
        raise Exception('error importing pyHalo, which is needed to use this routine')

    c = lens_cosmo.NFW_concentration(M, z, scatter=include_c_scatter)
    if include_c_scatter is False and c_scatter_add_dex > 0:
        c = 10 ** (np.log10(c) + c_scatter_add_dex)

    rhos, rs, _ = lens_cosmo.NFW_params_physical(M, c, z)
    halo_age = cosmo.halo_age(z, z_collapse)
    if halo_age < 0.:
        raise Exception('the halo age is negative, you have probably specified a collapse redshift < halo redshift')

    return solve_from_NFW_params(rhos, rs, halo_age, cross_section_type, kwargs_cross_section, **kwargs_solver)
