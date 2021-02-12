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

    def maxwell_boltzmann_average(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v * sigma(v) dv

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2), and v_p is most probable speed, and
        v_rms is the velocity dispersion (or the r.m.s. speed). in general v_mp^2 = 2/3 v_rms^2

        :param v_rms: the RMS speed (velocity dispersion) of the halo
        :return: the velocity-weighted cross section in units km/sec * cm^2/gram
        """
        most_probable_v = v_rms * np.sqrt(2. / 3)

        def _integrand(v):
            x = v / most_probable_v
            kernel = 4 * np.pi * v ** 3 * np.exp(-x ** 2)
            norm = (np.pi * most_probable_v ** 2) ** -1.5
            return norm * kernel * self.evaluate(v)

        return quad(_integrand, 0, 100 * most_probable_v)[0]
