from sidmpy.Solver.solver import solve_profile
from sidmpy.Solver.solution_interp.log_rho_interpolation import *
from sidmpy.CrossSections.power_law import PowerLaw
from sidmpy.CrossSections.tchannel import TChannel
from multiprocessing.pool import Pool
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

def single_solve(args):

    (halo_mass, halo_redshift, cross_section_class, function_params_physical, function_concentration,
    function_halo_age, kwargs_solver) = args
    c = function_concentration(halo_mass, halo_redshift, scatter=False)
    rhos, rs, _ = function_params_physical(halo_mass, c, halo_redshift)
    halo_age = function_halo_age(halo_redshift)
    try:
        rho0, sigma, r1 = solve_profile(rhos, rs, cross_section_class, halo_age, **kwargs_solver)
    except:
        rho0 = numpy.nan
    return rho0

def solve_array_power_law(filename_out, function_params_physical, function_concentration,
                 function_halo_age, kwargs_solver={}, nproc=10, i_start=0, i_end=-1):

    dim1, dim2, dim3, dim4 = len(cross_section_normalization_powerlaw), len(redshifts_powerlaw), len(mass_values_powerlaw), \
                             len(v_dependence_powerlaw)
    print('ntotal: ', dim1 * dim2 * dim3 * dim4)

    for v_dep in v_dependence_powerlaw[i_start:i_end]:
        args = []
        for cross_norm in cross_section_normalization_powerlaw:
            for zi in redshifts_powerlaw:
                for mi in mass_values_powerlaw:
                    kwargs_cross = {'norm': cross_norm, 'v_dep': v_dep, 'v_ref': 30.}
                    cross_section = PowerLaw(**kwargs_cross)
                    new = (10**mi, zi, cross_section, function_params_physical, function_concentration,
                                              function_halo_age, kwargs_solver)
                    args.append(new)

        pool = Pool(nproc)
        result = pool.map(single_solve, args)
        pool.close()
        result_array = numpy.array(result).reshape(dim1, dim2, dim3)
        result_array = numpy.log10(result_array)
        result_array = numpy.round(result_array, 4)
        with open(filename_out, 'a') as f:
            f.write('log_rho_vpower'+str(v_dep) +' = np.')
            f.write(str(repr(result_array))+ '\n\n')

def solve_array_tchannel(filename_out, function_params_physical, function_concentration,
                 function_halo_age, kwargs_solver={}, nproc=10, v_power=50):

    dim1, dim2, dim3 = len(cross_section_normalization_tchannel), len(redshifts_tchannel), len(mass_values_tchannel)
    dim4 = len(v_dependence_tchannel)
    print('ntotal: ', dim1 * dim2 * dim3 * dim4)

    #for v_dep in v_dependence_tchannel[i_start:i_end]:
    for v_dep in [v_power]:
        args = []
        for cross_norm in cross_section_normalization_tchannel:
            for zi in redshifts_tchannel:
                for mi in mass_values_tchannel:
                    kwargs_cross = {'norm': cross_norm, 'v_ref': v_dep}
                    cross_section = TChannel(**kwargs_cross)
                    new = (10**mi, zi, cross_section, function_params_physical, function_concentration,
                                              function_halo_age, kwargs_solver)
                    args.append(new)

        pool = Pool(nproc)
        result = pool.map(single_solve, args)
        pool.close()
        result_array = numpy.array(result).reshape(dim1, dim2, dim3)
        result_array = numpy.log10(result_array)
        result_array = numpy.round(result_array, 4)
        with open(filename_out, 'a') as f:
            f.write('log_rho_w'+str(v_dep)+' = np.')
            f.write(str(repr(result_array))+ '\n\n')

# from pyHalo.Halos.lens_cosmo import LensCosmo
# lc = LensCosmo()
# function_params_physical = lc.NFW_params_physical
# function_concentration = lc.NFW_concentration
# function_halo_age = lc.cosmo.halo_age
# kwargs_solver = {'rmin_profile': 1e-6}
# solve_array_tchannel('tchannel_solution_2850.py', function_params_physical, function_concentration,
#                  function_halo_age, kwargs_solver)
