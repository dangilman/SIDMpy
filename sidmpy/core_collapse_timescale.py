import numpy as np

def evolution_timescale(rho_s, rs, velocity_averaged_cross_section):

    """
    Evaluates the timescale for the evolution of SIDM profiles using the expression t_0^{-1} = ...
    after Equation 2 in Nishikawa et al. (2020) https://arxiv.org/pdf/1901.00499.pdf.

    Labelling the timescale t0, we would generally expect the following behavior for an object in the field:
    - From (roughly) t = 0  to t = 50 * t0 self interactions result in core expansion
    - From t = 50 - 350 t0 not much happens
    - From t0 = 350 t0 onwards the core begins to contract

    The core collapse timescale is shorter for subhalos, or tidally stripped objects. Rather than starting around
    t = 350 t_0, Nishikawa et al. (2020) say it starts around t = 25-50 t_0 depending on the amount of tidal stripping.
    (see also Samaeie et al. (2020), Kahlhoefer et al. (2019))

    :param rho_s: the central density normalization of the collisionless NFW profile of the same mass
    :param rs: the scale radius of the collisionless NFW profile of the same mass
    :param velocity_averaged_cross_section: the velocity-averaged scattering cross section in cm^2/gram
    :param scattering_velocity: the velocity at which to evaluate the possibly velocity-dependent cross section; for
    a velocity independent cross section this is redundant
    :return: the time in Gyr from the collapse time of a halo until when it should start to core collapse
    """

    G = 4.3e-6
    a = 4/np.sqrt(np.pi)
    v0 = np.sqrt(4 * np.pi * G * rho_s * rs ** 2)
    const = 2.136e-19  # to year^{-1}
    t_inverse = a * const * v0 * rho_s * velocity_averaged_cross_section
    return 1e-9 / t_inverse
