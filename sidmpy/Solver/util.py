from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np

def compute_k(rhos, cm2_per_gram_times_sigmav, age):

    # cm^2 * solar masses * km / (kpc^3 * gram * sec)
    const = 2.14e-10

    return const * rhos * cm2_per_gram_times_sigmav * age

def compute_r1(rhos, rs, v, cross_section_class, halo_age):

    """

    :param rhos: units solar mass / kpc^3
    :param rs: kpc
    :param sigma_v: km/sec
    :param cross_section_class: class cm^2/gram
    :param halo_age: units Gyr
    :return:
    """

    cm2_per_gram_times_sigmav = cross_section_class.maxwell_boltzmann_average(v)

    const = 2.14e-10
    k = const * rhos * cm2_per_gram_times_sigmav * halo_age

    roots = np.roots([1, 2, 1, -k])
    lam = np.real(np.max(roots[np.where(np.isreal(roots))]))

    return lam * rs

def ode_system(x, f):
    z1 = f[1]
    z2 = -2 * x ** -1 * f[1] - np.exp(f[0])
    return [z1, z2]

def integrate_profile(rho0, s0, r_s, r_1, rmax_fac=1.2, rmin_fac=0.01,
                      r_min=None, r_max=None):

    G = 4.3e-6  # units kpc and solar mass
    length_scale = np.sqrt(s0 ** 2 * (4 * np.pi * G * rho0) ** -1)

    if r_max is None:
        x_max = rmax_fac * r_1 / length_scale
    else:
        x_max = r_max/length_scale

    if r_min is None:
        x_min = r_s * rmin_fac / length_scale
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
    x = rmax * rs ** -1
    return 4*np.pi*rhos*rs**3 * (np.log(1+x) - x * (1+x) ** -1)

def nfwprofile_density(r, rhos, rs):
    x = r * rs ** -1
    return rhos * (x*(1+x)**2) ** -1

def isothermal_profile_mass(r_iso, rho_iso, rmax):

    m = 0
    ri = 0
    dr = r_iso[1] - r_iso[0]
    idx = 0

    while ri <= rmax:

        m += 4*np.pi*ri**2 * rho_iso[idx] * dr
        idx += 1
        ri = r_iso[idx]
        dr = r_iso[idx+1] - ri

    return m

def isothermal_profile_density(r, r_iso, rho_iso):

    interp = interp1d(r_iso, rho_iso)

    return interp(r)
