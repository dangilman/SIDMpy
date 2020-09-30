import numpy as np
from copy import copy

from sidmpy.Solver.util import *

from sidmpy.Solver.nfw_vdispersion_interp import interp_vdis_nfw

def _solve_iterative(rhonfw, rsnfw, cross_section_class, t_halo,
                     rmin_fac, rmax_fac,
                     N, plot=False, tol = 0.004,
                     s0min_scale=0.5, s0max_scale=1.5,
                     rhomin_scale=0.3, rhomax_scale=3.5):

    fit_quality = 100

    rhomin = rhonfw * rhomin_scale
    rhomax = rhonfw * rhomax_scale
    rho_range = np.log10(rhomax) - np.log10(rhomin)

    s0nfw = interp_vdis_nfw(rsnfw, rhonfw, rsnfw)
    s0min, s0max = s0nfw*s0min_scale,  s0nfw*s0max_scale

    s0_range = s0max - s0min

    core_size_unitsrs_last = 1e+8
    fit_quality_last = 1e+9

    iter_count = 1
    N_iter_max = 8

    k = 4
    Nvalues = [N]*k
    iter_step = [1]*k

    for i in range(k,N_iter_max):
        Nvalues.append(int(Nvalues[i-1] + 5 + i + 1))
        iter_step.append(i+2)

    while fit_quality > tol:

        rho0, s0, core_size_unitsrs, fit_quality, keywords = _solve(rhonfw, rsnfw, cross_section_class,
                                                                   t_halo, rmin_fac, rmax_fac,
                         rhomin, rhomax, s0min, s0max, Nvalues[iter_count-1], plot=plot)

        core_size_unitsrs = np.round(core_size_unitsrs, 2)

        new_rho_range = rho_range * iter_step[iter_count-1] ** -1

        rhomin = 10**(np.log10(rho0) - 0.5*new_rho_range)
        rhomax = 10**(np.log10(rho0) + 0.5*new_rho_range)

        new_s0_range = s0_range * iter_step[iter_count] ** -1
        s0min = max(1e-1, s0 - 0.5 * new_s0_range)
        s0max = s0 + 0.5 * new_s0_range

        if (core_size_unitsrs == core_size_unitsrs_last) and (fit_quality < 0.5*fit_quality_last):
            break

        core_size_unitsrs_last = copy(core_size_unitsrs)
        fit_quality_last = copy(fit_quality)

        if iter_count > len(Nvalues)-1:
            break

        iter_count += 1

        if iter_count >= N_iter_max:
            break

    if fit_quality > 0.1:
        rho0 = np.nan
        s0 = np.nan
        core_size_unitsrs = np.nan

    return rho0, s0, core_size_unitsrs, fit_quality, keywords

def _solve(rhonfw, rsnfw, cross_section_class, t_halo, rmin_fac, rmax_fac,
          rho_start, rho_end, s0_start, s0_end, N, plot=False):

    s0nfw = interp_vdis_nfw(rsnfw, rhonfw, rsnfw)
    rhocenter = [rhonfw]

    if plot: print('NFW profile velocity dispersion (at rs):', np.round(s0nfw, 2))

    percent = [int(N ** 2 * p) for p in [0.25, 0.5, 0.75]]
    logrhovals = np.linspace(np.log10(rho_start), np.log10(rho_end), N)
    s0vals = np.linspace(s0_start, s0_end, N)
    logrhoarr, s0arr = np.meshgrid(logrhovals, s0vals)

    coords = np.vstack([logrhoarr.ravel(), s0arr.ravel()]).T
    denarr = []
    marr = []
    pcount = 0

    for i in range(0, int(coords.shape[0])):
        if i in percent and plot:
            print(str(100 * percent[pcount] / N ** 2) + '% .... ')
            pcount += 1

        if coords[i,1] < 0:
            marr.append(np.nan)
            denarr.append(np.nan)
            continue

        r1 = compute_r1(rhonfw, rsnfw, coords[i, 1], cross_section_class, t_halo)
        r_iso, rho_iso = integrate_profile(10 ** coords[i, 0],
                                           coords[i, 1], rsnfw, r1, rmin_fac=rmin_fac,
                                           rmax_fac=rmax_fac)

        mass_nfw, mass_iso = nfwprofile_mass(rhonfw, rsnfw, r1), isothermal_profile_mass(r_iso, rho_iso, r1)
        nfw_den, sidm_den = nfwprofile_density(r1, rhonfw, rsnfw), isothermal_profile_density(r1, r_iso, rho_iso)

        mr = mass_nfw / mass_iso
        dr = nfw_den / sidm_den

        mass_pen = np.absolute(mr - 1)
        den_pen = np.absolute(dr - 1)

        marr.append(mass_pen)
        denarr.append(den_pen)

    denarr, marr = np.absolute(np.array(denarr).reshape(N, N)), \
                   np.absolute(np.array(marr).reshape(N, N))

    tot = marr + denarr

    minidx = np.argmin(tot.ravel())
    fit_quality = tot.ravel()[minidx]
    if plot: print('fit: ', tot.ravel()[minidx])
    rho0, s0 = 10 ** coords[minidx, 0], coords[minidx, 1]

    core_density_ratio = rhonfw / rho0

    rhocenter.append(rho0)
    if plot:

        import matplotlib.pyplot as plt

        e1, e2 = logrhovals[0], logrhovals[-1]

        extent = [e1, e2, s0vals[0], s0vals[-1]]
        plt.imshow(np.log10(tot), origin='lower',
                   aspect='auto', cmap='jet', extent=extent);
        plt.scatter(np.log10(rho0), s0, color='w', s=140, marker='x')
        ax = plt.gca()

        ax.annotate('core size (units rs):\n' + str(np.round(core_density_ratio, 3)),
                    xy=(0.6, 0.65), xycoords='axes fraction', fontsize=13)

        text = r'$\sigma = $' + str(cross_section_class.norm) + ' cm^2 g^-1' + \
                   '\n ' + r'$\sigma \propto v$' + '^' + str(-cross_section_class.v_pow)

        ax.annotate(text, xy=(0.6, 0.8), xycoords='axes fraction', fontsize=13)
        # plt.colorbar(label=r'$\log_{10}\left(\chi^2\right)$')

        ax.set_xlabel(r'$\log_{10}\left(\rho_0\right)$', fontsize=14)
        ax.set_ylabel(r'$\sigma_v \ \left[\rm{km} \rm{sec^-1} \right]$', fontsize=14)
        plt.show()
        a=input('continue')

    keywords = {'r1': r1, 'r_iso': r_iso, 'rho_iso': rho_iso,
                'rhonfw': rhonfw, 'rsnfw': rsnfw}

    return rho0, s0, core_density_ratio, fit_quality, keywords

