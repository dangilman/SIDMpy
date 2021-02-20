M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

arcsec = 2 * 3.14159 / 360 / 3600  # arc second in radian

G = 6.67384 * 10 ** (-11) * Mpc ** -3 * M_sun  # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

c = 299792458 * Mpc ** -1  # speed of light [Mpc s^-1]

density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3

import numpy as np
from scipy.integrate import quad

def TNFWprofile(r, rhos, rs, rt):

    return coreTNFWprofile(r, rhos, rs, rt, 0.)

def coreTNFWprofile(r, rhos, rs, rt, rc, a=10):

    x = r / rs
    tau = rt / rs
    beta = rc / rs

    truncation_factor = tau ** 2 / (tau ** 2 + x ** 2)
    core_factor = (x ** a + beta ** a) ** (-1/a)

    return rhos * truncation_factor * core_factor * (1 + x) ** -2

def NFW_params_physical(M, c, z, astropy, density_threshold=200):

    h = astropy.h
    a_z = 1/(1 + z)

    rhoc = astropy.critical_density0.value * density_to_MsunperMpc / h ** 2
    r200_mpc = (3 * M / (4 * 3.14159 * rhoc * density_threshold)) ** (1. / 3.) / h * a_z # physical radius r200
    rho0_msun_mpc3 = density_threshold / 3 * rhoc * c ** 3 / (np.log(1 + c) - c / (1 + c)) # physical density in M_sun/Mpc**3
    Rs_mpc = r200_mpc / c

    return rho0_msun_mpc3 * 1000 ** -3, Rs_mpc * 1000

def pjaffe_Mr(R, rho, rt, ra):

    f = (ra * np.arctan(R/ra) - rt * np.arctan(R/rt)) / (ra ** 2 - rt ** 2)
    return 4 * np.pi * ra ** 2 * rt ** 2 * rho * f

def rho_pjaffe(m, rs, ra, r200):

    f = (ra * np.arctan(r200 / ra) - rs * np.arctan(r200 / rs)) / (ra ** 2 - rs ** 2)
    rho = m / (4 * np.pi * ra ** 2 * rs ** 2 * f)
    return rho

def pjaffe(r, m, c, rs, ra):
    """
    sigma0 = pi * rho0 * ra * rs / (rs + ra)

    Mtotal = 2 * np.pi * sigma0 * Ra * Rs = 2 * pi^2 * rho0 * ra^2 * rs^2 / (rs + ra)
    rho0 = M * (rs + ra) / (2 * pi^2 * ra^2 * rs^2)
    sigma0 = M / (2*pi * ra * rs)
    """

    rho0 = rho_pjaffe(m, rs, ra, c * rs)
    x1 = r/ra
    x2 = r/rs
    return rho0 / ((1 + x1 ** 2) * (1 + x2 ** 2))

def mean_density_inside_R(R, profile, profile_args):

    v = 4 * np.pi * R ** 3 / 3
    return total_mass(R, profile, profile_args) / v

def total_mass(R, profile, profile_args):

    def _integrand(x):
        return 4 * np.pi * x ** 2 * profile(x, *profile_args)

    return quad(_integrand, 0, R)[0]

