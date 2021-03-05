import numpy as np
from sidmpy.Solver.util import nfw_velocity_dispersion, compute_r1, compute_rho_sigmav_grid

def solve_profile(rho_s, rs, cross_section_class, halo_age, rmin_profile=0.001, rmax_profile=2.5,
                  vdis_min_scale=0.3, vdis_max_scale=2., rho_min_scale=0.04, rho_max_scale=10., plot=False, tol=1e-2,
                  solver_resolution=12, n_iter_max=12):

    """
    This function finds a solution to system of equations:
    1) rho_nfw(r_1) * <sigma * v> * t_halo  = 1
    2) M_nfw(r_1) = M_iso(r_1)
    3) rho_nfw(r_1) = rho_iso(r_1)
    where rho_nfw(r) is the density profile of an NFW halo, <sigma * v> is the velocity averaged cross section,
    t_halo is the age of the halo since it collapsed, M_nfw/M_iso is the mass enclosed for the NFW/isothermal
    profiles, and rho_iso(r) is the isothermal densiyt profile as a function of r.

    :param rho_s: density normalization of NFW halo in M_sun / kpc^3
    :param rs: scale radius of NFW halo in kpc
    :param cross_section_class: an instance of a cross section class (see CrossSections)
    :param halo_age: time since halo collapsed in Gyr
    :param rmin_profile: the minimum radius where the isothermal profile is computed (in units of rs)
    :param rmax_profile: the maximum radius where the isothermal profile is computed (in units of rs)
    :param vdis_min_scale: sets minimum velocity dispersion to initialize the search for a solution to Eq 1, 2, 3
    :param vdis_max_scale: sets maximum velocity dispersion to initialize the search for a solution to Eq 1, 2, 3
    The initial guess for the velocity dispersion will computed from the un-cored NFW profile
    :param rho_min_scale: sets the minimum core density used to initialize the search for a solution
    :param rho_max_scale: sets the max cored density used to initialize the search for a solution
    :param plot: If True, will produce a plot of the log(rho), velocity dispersion as the algorithm iterates
    :param tol: the tolerance for a solution to the boundary conditions 2 and 3
    :param solver_resolution: The number of grid points used to intialize a search for a solution
    :param n_iter_max: maximum number of iterations before quitting
    :return: the central density and central velocity dispersion of the SIDM profile that satisfies
    equations 1, 2, 3
    """
    rhomin = rho_s * rho_min_scale
    rhomax = rho_s * rho_max_scale
    logrhomin, logrhomax = np.log10(rhomin), np.log10(rhomax)
    log_rho_range = np.log10(rhomax) - np.log10(rhomin)

    vdis_init = nfw_velocity_dispersion(rs, rho_s, rs)
    vdismin, vdismax = vdis_init * vdis_min_scale, vdis_init * vdis_max_scale
    vdis_range = vdismax - vdismin

    n_iter = 0

    while True:

        if n_iter > 1:
            log_rho_range *= 0.6
            vdis_range *= 0.6
            solver_resolution *= 1.3
            solver_resolution = min(20, int(solver_resolution))

        _x = np.linspace(logrhomin, logrhomax, solver_resolution)
        _y = np.linspace(vdismin, vdismax, solver_resolution)
        log_rho_values, vdis_values = np.meshgrid(_x, _y)
        log_rho_values = log_rho_values.ravel()
        vdis_values = vdis_values.ravel()

        fit_grid = compute_rho_sigmav_grid(log_rho_values, vdis_values, rho_s, rs, cross_section_class,
                                           halo_age, rmin_profile, rmax_profile)

        idx_best = np.argmin(fit_grid)
        fit_quality_last = fit_grid[idx_best]
        log_rho_best = log_rho_values[idx_best]
        vdis_best = vdis_values[idx_best]

        if plot:
            import matplotlib.pyplot as plt
            grid = np.log10(fit_grid.reshape(solver_resolution, solver_resolution))
            aspect = abs(logrhomax - logrhomin)/(vdismax - vdismin)
            worst = np.max(grid)
            best = np.min(grid)
            plt.clf()
            fig = plt.figure(1)
            fig.set_size_inches(5.5, 5.5)
            ax = plt.gca()
            ax.imshow(grid, extent=[logrhomin, logrhomax, vdismin, vdismax],
                       aspect=aspect, cmap='bwr', origin='lower', vmin=best, vmax=worst)
            ax.annotate('ITERATION #'+str(n_iter+1), xy=(0.6, 0.92), xycoords='axes fraction', fontsize=14)
            ax.annotate(r'$\log_{10}\left(\rho_0\right) = $'+str(np.round(log_rho_best, 2)), xy=(0.6, 0.84),
                        xycoords='axes fraction', fontsize=14)
            ax.annotate(r'$\sigma_{v} = $' + str(np.round(vdis_best, 2)), xy=(0.6, 0.76),
                        xycoords='axes fraction', fontsize=14)
            ax.annotate('fit quality: ' + str(np.round(fit_quality_last, 3)), xy=(0.04, 0.9),
                        xycoords='axes fraction', fontsize=14)
            ax.scatter(log_rho_best, vdis_best, color='k', s=45, marker='*')
            ax.set_xlabel(r'$\log_{10} \left(\rho_0\right)$', fontsize=14)
            ax.set_ylabel(r'$\sigma_v \left[\rm{km} \rm{s^{-1}}\right]$', fontsize=14)
            plt.show()
            a=input('continue with next iteration... ')

        logrhomin = log_rho_best - 0.5 * log_rho_range
        logrhomax = log_rho_best + 0.5 * log_rho_range
        vdismin = max(0.1, vdis_best - vdis_range)
        vdismax = vdis_best + vdis_range

        if fit_quality_last < tol or n_iter > n_iter_max:
            break
        else:
            n_iter += 1

    r1 = compute_r1(rho_s, rs, vdis_best, cross_section_class, halo_age)
    return 10 ** log_rho_best, vdis_best, r1
