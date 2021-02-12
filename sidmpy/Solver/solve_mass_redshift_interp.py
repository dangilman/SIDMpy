from sidmpy.Solver.solver import solve_profile
from sidmpy.Solver.solution_interp.log_rho_interpolation import cross_section_normalization, redshifts, mass_values, v_dependence_powerlaw
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
                 function_halo_age, kwargs_solver={}, nproc=10):

    dim1, dim2, dim3, dim4 = len(cross_section_normalization), len(redshifts), len(mass_values), len(v_dependence_powerlaw)
    print('ntotal: ', dim1 * dim2 * dim3 * dim4)

    for v_dep in v_dependence_powerlaw:
        args = []
        for cross_norm in cross_section_normalization:
            for zi in redshifts:
                for mi in mass_values:
                    kwargs_cross = {'norm': cross_norm, 'v_dep': v_dep, 'v_ref': 30.}
                    cross_section = PowerLaw(**kwargs_cross)
                    new = (10**mi, zi, cross_section, function_params_physical, function_concentration,
                                              function_halo_age, kwargs_solver)
                    args.append(new)

        pool = Pool(nproc)
        result = pool.map(single_solve, args)
        pool.close()
        result_array = numpy.array(result).reshape(dim1, dim2, dim3)
        with open(filename_out, 'a') as f:
            f.write('log_rho_vpower'+str(v_dep) +' = numpy.')
            f.write(str(repr(result_array))+ '\n\n')

def solve_array_tchannel(filename_out, function_params_physical, function_concentration,
                 function_halo_age, kwargs_solver={}, nproc=10):

    dim1, dim2, dim3 = len(cross_section_normalization), len(redshifts), len(mass_values)
    print('ntotal: ', dim1 * dim2 * dim3)
    args = []
    for cross_norm in cross_section_normalization:
        for zi in redshifts:
            for mi in mass_values:
                kwargs_cross = {'norm': cross_norm, 'v_ref': 30.}
                cross_section = TChannel(**kwargs_cross)
                new = (10**mi, zi, cross_section, function_params_physical, function_concentration,
                                          function_halo_age, kwargs_solver)
                args.append(new)

    pool = Pool(nproc)
    result = pool.map(single_solve, args)
    pool.close()
    result_array = numpy.array(result).reshape(dim1, dim2, dim3)
    with open(filename_out, 'a') as f:
        f.write('log_rho_w30 = numpy.')
        f.write(str(repr(result_array))+ '\n\n')
#
# from pyHalo.Halos.lens_cosmo import LensCosmo
# lc = LensCosmo()
# function_params_physical = lc.NFW_params_physical
# function_concentration = lc.NFW_concentration
# function_halo_age = lc.cosmo.halo_age
# kwargs_solver = {'rmin_profile': 1e-6}
# solve_array_power_law('powerlaw_solution.py', function_params_physical, function_concentration, function_halo_age)
# solve_array_tchannel('tchannel_solution.py', function_params_physical, function_concentration, function_halo_age)
