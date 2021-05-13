from scipy.integrate import solve_ivp, quad
from sidmpy.Profiles.halo_density_profiles import TNFWprofile
from scipy.interpolate import interp1d
import numpy as np
from scipy.optimize import fsolve

def compute_r1(rhos, rs, vdispersion_halo, cross_section_class, halo_age):

    """

    :param rhos: density normalization of NFW profile in M_sun / kpc^3
    :param rs: scale radius in kpc
    :param vdispersion_halo: the central velocity dispersion of the halo in km/sec
    :param cross_section_class: an instance of a cross section class (currently the only option implemented is PowerLaw)
    :param halo_age: units Gyr
    :return: the solution to the equation for r_1 in kpc:
    rho_nfw(r_1) * <sigma * v> * t_halo = 1
    """
    # units cm^2 / gram * km/sec
    cm2_per_gram_times_sigmav = cross_section_class.scattering_rate_cross_section(vdispersion_halo)

    # cm^2 * solar masses * km / (kpc^3 * gram * sec) to 1/Gyr
    const = 2.1358e-10

    k = rhos * cm2_per_gram_times_sigmav * halo_age # cm^2 / g * km/sec * time * M_sun / kpc^3
    k *= const # to a dimensionless number
    roots = np.roots([1, 2, 1, -k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))

    return lam * rs

def compute_r1_nfw_velocity_dispersion(rhos, rs, cross_section_class, halo_age):

    # cm^2 * solar masses * km / (kpc^3 * gram * sec) to 1/Gyr
    const = 2.1358e-10

    def _func_to_min(r):
        r = r[0]
        vdispersion_halo = nfw_velocity_dispersion_analytic(r, rhos, rs)
        cm2_per_gram_times_sigmav = cross_section_class.scattering_rate_cross_section(vdispersion_halo)
        func = const * TNFWprofile(r, rhos, rs, 100000 * rs) * cm2_per_gram_times_sigmav * halo_age - 1
        return func

    out = fsolve(_func_to_min, rs)

    return out[0]

def ode_system(x, f):
    """
    Decomposes the second order ODE into two first order ODEs
    """
    z1 = f[1]
    z2 = -2 * x ** -1 * f[1] - np.exp(f[0])
    return [z1, z2]

def integrate_profile(rho0, s0, r_s, r_1, rmax_fac=1.2, rmin_fac=0.01,
                      r_min=None, r_max=None):

    """
    Solves the ODE describing the to obtain the density profile

    :returns: the integration domain in kpc and the solution to the density profile in M_sun / kpc^3
    """
    G = 4.3e-6  # units kpc and solar mass
    length_scale = np.sqrt(s0 ** 2 * (4 * np.pi * G * rho0) ** -1)

    if r_max is None:
        x_max = rmax_fac * r_1 / length_scale
    else:
        x_max = r_max/length_scale

    if r_min is None:
        x_min = r_1 * rmin_fac / length_scale
    else:
        x_min = r_min/length_scale

    # solve the ODE with initial conditions
    phi_0, phi_prime_0 = 0, 0
    N = 600

    xvalues = np.linspace(x_min, x_max, N)

    res = solve_ivp(ode_system, (x_min, x_max),
                    [phi_0, phi_prime_0], t_eval=xvalues)

    return res['t'] * length_scale, rho0 * np.exp(res.y[0])

def nfwprofile_mass(rhos, rs, rmax):
    """
    Computes the mass of an NFW profile inside R = rmax
    """
    x = rmax * rs ** -1
    return 4*np.pi*rhos*rs**3 * (np.log(1+x) - x * (1+x) ** -1)

def isothermal_profile_mass(r_iso, rho_iso, rmax):
    """
    Integrates the isothermal profile density out to a radius rmax
    """

    mass = 0
    dr_step = r_iso[1] - r_iso[0]
    count = 0
    assert r_iso[-1] > rmax
    while True:
        mass += 4 * np.pi * r_iso[count] ** 2 * rho_iso[count] * dr_step
        count += 1
        if r_iso[count] > rmax:
            break

    return mass

def isothermal_profile_density(r, r_iso, rho_iso):
    """
    Evalutes the density of the isothermal mass profile at a radius r
    """
    rho_iso_interp = interp1d(r_iso, rho_iso,
                              fill_value=(rho_iso[0], 0.), bounds_error=False)
    return rho_iso_interp(r)

def nfw_velocity_dispersion_fromfit(m):
    """
    The velocity dispersion of an NFW profile with mass m calibrated from a power law fit for halos
    between 10^6 and 10^10 at z=0
    :param m: halo mass in M_sun
    :return: the velocity dispersion inside rs
    """
    coeffs = [0.31575757, -1.74259129]
    log_vrms = coeffs[0] * np.log10(m) + coeffs[1]
    return 10 ** log_vrms

def nfw_mass_from_velocity_dispersion(vrms):

    """
    The velocity dispersion of an NFW profile with mass m calibrated from a power law fit for halos
    between 10^6 and 10^10 at z=0
    :param m: halo mass in M_sun
    :return: the velocity dispersion inside rs
    """
    coeffs = [0.31575757, -1.74259129]
    log_vmrs = np.log10(vrms)
    logm = (log_vmrs - coeffs[1])/coeffs[0]
    return 10 ** logm

def nfw_circular_velocity(r, rhos, rs):

    G = 4.3e-6
    x = r/rs
    fx = np.log(1+x) -x/(1+x)
    m = 4 * np.pi * rs ** 3 * rhos * fx
    return np.sqrt(G * m / r)

def nfw_velocity_dispersion(r, rho_s, rs, tol=1e-4):
    """

    :param r:
    :param rho_s:
    :param rs:
    :return:
    """
    def _integrand(rprime):
        return TNFWprofile(rprime, rho_s, rs, 1000000000 * rs) * nfwprofile_mass(rho_s, rs, rprime) / rprime ** 2

    rmax_init = 2 * rs
    rmax_scale = 2.
    rmax = rmax_init + rmax_scale * rs
    integral = quad(_integrand, r, rmax)[0]
    count_max = 5.
    count = 0
    while True:
        rmax *= rmax_scale
        integral_new = quad(_integrand, r, rmax)[0]
        fit = abs(integral_new/integral - 1)

        if fit < tol:
            break
        elif count > count_max:
            print('warning: NFW velocity dispersion computtion did not converge to more than '+str(fit))
            break
        else:
            count += 1
            integral = integral_new

    G = 4.3e-6  # units kpc/M_sun * (km/sec)^2
    sigma_v_squared = G * integral_new / TNFWprofile(r, rho_s, rs, 1e+6 * rs)
    return np.sqrt(sigma_v_squared)

def Li(x):
    integrand = lambda u: np.log(1 - u) / u
    return quad(integrand, x, 0)[0]

def nfw_velocity_dispersion_analytic(r, rhos, rs):

    G = 4.3e-6
    x = r / rs
    factor = 0.5 * x * (1 + x) ** 2 * G * 4 * np.pi * rhos * rs ** 2

    term = np.pi ** 2 - np.log(x) - 1 / x - 1 / (1 + x) ** 2 - 6 / (1 + x) + \
           (1 + 1 / x ** 2 - 4 / x - 2 / (1 + x)) * np.log(1 + x) + 3 * np.log(1 + x) ** 2 + 6 * Li(-x)

    return np.sqrt(factor * term)

def compute_rho_sigmav_grid(log_rho_values, vdis_values, rhos, rs, cross_section_class,
                            halo_age, rmin_profile, rmax_profile, use_nfw_velocity_dispersion):

    fit_grid = np.ones_like(log_rho_values) * 1e+12
    fit_grid = fit_grid.ravel()
    # compute the fit quality for each point in the search space

    for i, (log_rho_i, velocity_dispersion_i) in enumerate(zip(log_rho_values, vdis_values)):

        if use_nfw_velocity_dispersion:
            r1 = compute_r1_nfw_velocity_dispersion(rhos, rs, cross_section_class, halo_age)
        else:
            r1 = compute_r1(rhos, rs, velocity_dispersion_i, cross_section_class, halo_age)

        r_iso, rho_iso = integrate_profile(10 ** log_rho_i,
                                           velocity_dispersion_i, rs, r1, rmin_fac=rmin_profile,
                                           rmax_fac=rmax_profile)

        m_enclosed = isothermal_profile_mass(r_iso, rho_iso, r1)
        rho_at_r1 = isothermal_profile_density(r1, r_iso, rho_iso)
        m_nfw = nfwprofile_mass(rhos, rs, r1)
        rho_nfw_at_r1 = TNFWprofile(r1, rhos, rs, 10000000 * rs)

        mass_ratio = m_nfw / m_enclosed
        density_ratio = rho_nfw_at_r1 / rho_at_r1

        mass_penalty = np.absolute(mass_ratio - 1)
        den_penalty = np.absolute(density_ratio - 1)

        fit_qual = mass_penalty + den_penalty
        fit_grid[i] = fit_qual

    return fit_grid
