from scipy.integrate import quad
import numpy as np

class PowerLaw(object):

    def __init__(self, cross0, v_ref=30, v_dep=0):

        self.norm = cross0

        if v_dep < 0:
            raise Exception('you have specified v_dep = '+str(v_dep) + ', this is unphysical.')
        self.v_pow = v_dep
        self.v_ref = v_ref

    def maxwell_boltzmann_average(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v * sigma(v) dv

        where K(v) is the Maxwell Boltzmann kernel: 4*pi*v^2*exp(-v^2/v_p^2)
        where v_p is most probable speed

        """
        most_probable_v = v_rms * np.sqrt(2. / 3)

        # 2 * np.sqrt(2./3) / np.sqrt(np.pi)
        # cross section times v_avg = cross0 * above factor

        func = self.sigma

        def _integrand(v):
            x = v * most_probable_v ** -1
            kernel = 4 * np.pi * v ** 3 * np.exp(-x ** 2)
            norm = (np.pi * most_probable_v ** 2) ** -1.5

            return norm * kernel * func(v)

        integral = quad(_integrand, 0, 10 * most_probable_v)[0]

        return integral

    def sigma(self, v):

        x = self.v_ref * v ** -1

        return self.norm * x ** self.v_pow
