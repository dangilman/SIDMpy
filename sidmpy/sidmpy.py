from sidmpy.CrossSections.power_law import PowerLaw
from sidmpy.Solver.solve_isothermal import _solve_iterative
from sidmpy.Profiles.nfw import NFW_params_physical
import numpy as np

def SIDMrho(rhos, rs, t_halo, cross_norm, v_dep,
                    plot=False, N_iter=5, rmin_fac=0.01, rmax_fac=1.4, tol=0.002,
            s0min_scale=0.5, s0max_scale=2., rhomin_scale=0.5, rhomax_scale=4):

    """

    This function computes the central density of an SIDM halo using a relatively simple Jeans argument first
    introduced by Kaplinghat et al. 2016, and recently tested by Robertson et al. 2020. This function assumes a
    specific form for the SIDM interaction cross section:

    sigma(v) = cross_norm * (v / 30) ^ v_dep

    i.e. it is a velocity dependent cross section that pivots around 30 km/sec, a typical velocity dispersion
    for a dwarf galaxy. "cross_norm" in this context is the cross section in units cm^2 / gram at 30 km/sec, but you can
    easily do computations for a velocity independent cross section by setting v_dep = 0.

    :param rhos: central density of NFW halo [M_sun / kpc]
    rho_NFW(x) = rhos / (x * (1+x)^2); x = r/rs

    :param rs: scale radius of NFW halo [kpc]
    :param cross_norm: normalization of SIDM cross section at 30 km/sec [cm^2 / gram]
    :param v_dep: velocity dependence of cross section (must be between 0 and 1), negative values are unphysical
    :param t_halo: the age of the halo in Gyr
    :param plot: If True, it will show plots of the velocity dispersion/rho_central parameter space as it finds a solution
    to the Jeans equation
    :param N_iter: number of iterations for the solver
    :param rmin_fac: looks for solutions to r_1 > rmin_fac * rs
    :param rmax_fac: looks for solutions to r_1 < rmax_fac * rs
    :param tol: tolerance for the boundary conditions imposed on the Jeans analysis
    :param s0min_scale: same as rmin_fac, but for the central velocity dispersion relative to the velocity dispersion
    of the reference CDM halo
    :param s0min_scale: same as rmax_fac, but for the central velocity dispersion relative to the velocity dispersion
    of the reference CDM halo
    :param rhomin_scale: same as rmin_fac, but for the central density of the SIDM core relative to rhos
    :param rhomax_scale: same as rmax_fac, but for the central density of the SIDM core relative to rhos

    RETURNS:

    rho0: the central core density of the SIDM halo

    s0: the velocity dispersion of the SIDM halo

    core_size_unitsrs: a natural definition of the core size, measured in units of the scale radius
    core_size_unitsrs = r_core / rs = rhos / rho0

    fit_quality: how well the boundary conditions are met
    keywords: all the info you would possibly want (probably redundant)
    """

    cross_section_class = PowerLaw(cross_norm, v_dep=v_dep)

    rho0, s0, core_size_unitsrs, fit_quality, keywords = _solve_iterative(
        rhos, rs, cross_section_class, t_halo, rmin_fac, rmax_fac,
        N_iter, plot=plot, tol=tol,
        s0min_scale=s0min_scale, s0max_scale=s0max_scale,
        rhomin_scale=rhomin_scale, rhomax_scale=rhomax_scale,
    )

    if fit_quality > 0.1:
        print('SOLVING FOR THE CENTRAL DENSITY HAS FAILED.')
        rho0, s0, core_size_unitsrs, fit_quality = np.nan, np.nan, np.nan, np.nan

    return rho0, s0, core_size_unitsrs, fit_quality, keywords

def SIDMrho_fromMz(m200, z, z_collapse, cross_norm, v_dep, c=None, plot=False, N_iter=5, rmin_fac=0.01, rmax_fac=1.4, tol=0.002,
                   astropy=None, kwargs_cosmo=None, s0min_scale=0.5, s0max_scale=2., rhomin_scale=0.5, rhomax_scale=4):

    """
    This function does exactly the same thing as SIDMrho, but the input arguments are mass, redshift, and z_collapse
    rather than (rhos, rs, t_halo). Here, z_collapse is the redshift at which the halo formed (typically z~10 for
    sub-galactic scales).

    In order to use this function you need to install colossus http://www.benediktdiemer.com/code/colossus/
    to evaluate the concentration-mass relation.

    :param m200: halo mass, computed with respect to 200*rho_crit at z=0. Note that the code expects a physical mass,
    no "little h" nonsense [M_sun]
    :param z: redshift of the halo

    see SIDMrho function for the rest of the documentation

    """

    try:

        if astropy is None:
            if kwargs_cosmo is None:
                kwargs_cosmo = {'H0': 70, 'Om0': 0.235 + 0.0464, 'Ob0': 0.0464, 'ns': 0.9608,
                                'sigma8': 0.82}

            from astropy.cosmology import FlatLambdaCDM
            astropy = FlatLambdaCDM(H0=kwargs_cosmo['H0'], Om0=kwargs_cosmo['Om0'], Ob0=kwargs_cosmo['Ob0'])

        else:
            kwargs_cosmo = {'H0': astropy.h * 100, 'Om0': astropy.Om0, 'Ob0': astropy.Ob0, 'ns': 0.9608,
                            'sigma8': 0.82}

    except:
        raise Exception('must install astropy to use this function')

    if c is None:
        try:
            from colossus.cosmology import cosmology
            from colossus.halo.concentration import concentration

            _ = cosmology.setCosmology('custom', kwargs_cosmo)

        except:
            raise Exception('must install colossus to use this function without specifying a concentration: '
                            'http://www.benediktdiemer.com/code/colossus/')

    halo_age = astropy.age(z).value - astropy.age(z_collapse).value
    assert halo_age > 0

    if c is None:
        c = concentration(m200 * astropy.h, mdef='200c', model='diemer19', z=z)

    rhos, rs = NFW_params_physical(m200, c, z, astropy)

    return SIDMrho(rhos, rs, halo_age, cross_norm, v_dep, plot, N_iter, rmin_fac, rmax_fac, tol,
                   s0min_scale, s0max_scale, rhomin_scale, rhomax_scale)
