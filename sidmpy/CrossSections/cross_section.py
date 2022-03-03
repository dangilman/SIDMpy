import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

class InteractionCrossSection(object):

    """
    This is a general class for an interaction cross section that computes properties of the cross section,
    particularly integrals over it weighted by velocity to some power.

    Each specific cross section is a subclass of this this class, and should have a (private) function called
    _velocity_dependence_kernel that returns the amplitude of the cross section given a velocity.
    """

    def __init__(self, norm, velocity_dependence_kernel):

        self.norm = norm
        self._vdep_func = velocity_dependence_kernel
        self._scattering_rate_cross_section_interp = None

    def interpolate_scattering_rate_cross_section(self, vmin=0.1, vmax=2000):

        log10v = np.linspace(np.log10(vmin), np.log10(vmax), 300)
        sigma_v = self.scattering_rate_cross_section(10 ** log10v)
        self._scattering_rate_cross_section_interp = interp1d(log10v, sigma_v)

    def evaluate(self, v):
        """
        Evaluates the scattering crossing section at a particular speed
        """

        return self.norm * self._vdep_func(v)

    def velocity_weighted_average(self, v_rms, n):

        """
        Evaluates <v^n sigma(v)>
        :return:
        """
        return self._integral(v_rms, n, self.evaluate)

    def velocity_moment(self, v_rms, n):

        """
        Computes the velocity moment of the MB distribution <v^n>
        :param v_rms:
        :return:
        """

        velocity_dependence_function = lambda x: 1.
        return self._integral(v_rms, n, velocity_dependence_function)

    def velocity_weighted_cross_section(self, v_rms):
        """
        Averages the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        <sigma(v) v> / <v>
        """

        return self.velocity_weighted_average(v_rms, 1) / self.velocity_moment(1)

    def scattering_rate_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution to compute
        <sigma(v) v>:

        \int_{0}^{infinity} K(v) * v * sigma(v) dv

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), and v_p is the
        most probable speed related to the velocity dispersion v_rms by v_p^2 = 2/3 v_rms^2

        :param v_rms: the RMS speed (velocity dispersion) of the halo
        :return: the velocity-weighted cross section in units km/sec * cm^2/gram
        """

        if self._scattering_rate_cross_section_interp is None:

            if isinstance(v_rms, list) or isinstance(v_rms, np.ndarray):
                integral = [self._integral(vi, 1, self.evaluate) for vi in v_rms]
                return np.array(integral)
            else:
                return self._integral(v_rms, 1, self.evaluate)

        else:
            return self._scattering_rate_cross_section_interp(np.log10(v_rms))

    def momentum_exchange_average(self, v_rms):

        return self.velocity_weighted_average(v_rms, 2)

    def momentum_transfer_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^2 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        return self.momentum_exchange_average(v_rms)/self.velocity_moment(v_rms, 2)

    def energy_transfer_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^2 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        return self.velocity_weighted_average(v_rms, 3)/self.velocity_moment(v_rms, 3)

    def _integral(self, v_rms, n, func):

        """

        :param v_rms: the r.m.s. velocity dispersion of the halo
        :param n: the exponent of the velocity term in the integrand
        :param func: a function that returns the cross section as a function of velocity
        :return: the integral in Equation 4 of this paper https://arxiv.org/pdf/2102.09580.pdf

        K(v) * v^n * sigma(v)

        where K(v) is the Maxwell Boltzmann kernel

        """
        args = (v_rms, n, func)
        return quad(self._integrand, 0, 100 * v_rms, args)[0]

    @staticmethod
    def _integrand(v, v_rms, n, func):
        x = v ** 2 / (4 * v_rms ** 2)
        kernel = v ** (2 + n) * np.exp(-x) / (2 * np.sqrt(np.pi) * v_rms ** 3)
        return kernel * func(v)



