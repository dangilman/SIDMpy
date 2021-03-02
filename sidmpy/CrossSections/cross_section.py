import numpy as np
from scipy.integrate import quad

class InteractionCrossSection(object):

    def __init__(self, norm, velocity_dependence_kernel):

        self.norm = norm
        self._vdep_func = velocity_dependence_kernel

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

        return self._integral(v_rms, 1, self.evaluate)

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

        args = (v_rms, n, func)
        return quad(self._integrand, 0, 100 * v_rms, args)[0]

    @staticmethod
    def _integrand(v, v_rms, n, func):
        x = v ** 2 / (4 * v_rms ** 2)
        kernel = v ** (2 + n) * np.exp(-x) / (2 * np.sqrt(np.pi) * v_rms ** 3)
        return kernel * func(v)


