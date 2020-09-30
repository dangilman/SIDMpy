M_sun = 1.9891 * 10 ** 30  # solar mass in [kg]

Mpc = 3.08567758 * 10 ** 22  # Mpc in [m]

arcsec = 2 * 3.14159 / 360 / 3600  # arc second in radian

G = 6.67384 * 10 ** (-11) * Mpc ** -3 * M_sun  # Gravitational constant [Mpc^3 M_sun^-1 s^-2]

c = 299792458 * Mpc ** -1  # speed of light [Mpc s^-1]

density_to_MsunperMpc = 0.001 * M_sun**-1 * (100**3) * Mpc**3

import numpy as np

def TNFWprofile(r, rhos, rs, rt):

    return coreTNFWprofile(r, rhos, rs, rt, 0.)

def coreTNFWprofile(r, rhos, rs, rt, rc, a=10):

    x = r / rs
    tau = rt / rs
    beta = rc/rs

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
