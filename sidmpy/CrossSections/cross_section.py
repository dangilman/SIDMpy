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

    def velocity_weighted_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * sigma(v) dv

        K(v) is the Maxwell Boltzmann kernel:
        4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2)

        where v_p is the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        return self._integral(v_rms, 0., self.evaluate)

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

    def momentum_transfer_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^2 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        velocity_dependence_function = lambda x: 1.
        return self._integral(v_rms, 2, self.evaluate) / self._integral(v_rms, 2, velocity_dependence_function)

    def _integral(self, v_rms, n, func):

        most_probable_v = v_rms * np.sqrt(2. / 3)
        args = (most_probable_v, n, func)
        return quad(self._integrand, 0, 100 * most_probable_v, args)[0]

    @staticmethod
    def _integrand(v, v0, n, func):
        x = v / v0
        kernel = 4 * np.pi * v ** (2 + n) * np.exp(-x ** 2)
        norm = (np.pi * v0 ** 2) ** -1.5
        return norm * kernel * func(v)


