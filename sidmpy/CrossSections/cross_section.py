import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sidmpy.core_collapse_timescale import fraction_collapsed_halos

class InteractionCrossSection(object):

    """
    This is a general class for an interaction cross section that computes properties of the cross section,
    particularly integrals over it weighted by velocity to some power.

    Each specific cross section is a subclass of this this class, and should have a (private) function called
    _velocity_dependence_kernel that returns the amplitude of the cross section given a velocity.
    """

    def __init__(self, norm, velocity_dependence_kernel,
                 vmin_integral=None, vmax_integral=None, use_trap_z=False, auto_interp=False,
                 vmin_auto_interp=0.1, vmax_auto_interp=350, n_steps=250):

        self.norm = norm
        self._vdep_func = velocity_dependence_kernel

        self.vmin_integral = vmin_integral
        self.vmax_integral = vmax_integral
        self.use_trap_z = use_trap_z

        self._scattering_rate_cross_section_interp = None
        self._energy_transfer_cross_section_interp = None
        self._v5_cross_interp = None

        if auto_interp:
            #self.interpolate_scattering_rate_cross_section(vmin_auto_interp, vmax_auto_interp, n_steps)
            #self.interpolate_energy_transfer_cross_section(vmin_auto_interp, vmax_auto_interp, n_steps)
            self.interpolate_v5_cross_section(vmin_auto_interp, vmax_auto_interp, n_steps)

    def interpolate_scattering_rate_cross_section(self, vmin, vmax, n_steps):

        log10v = np.linspace(np.log10(vmin), np.log10(vmax), n_steps)
        log10sigma_v = np.log10(self.scattering_rate_cross_section(10 ** log10v))
        self._scattering_rate_cross_section_interp = interp1d(log10v, log10sigma_v)

    def interpolate_energy_transfer_cross_section(self, vmin, vmax, n_steps):

        log10v = np.linspace(np.log10(vmin), np.log10(vmax), n_steps)
        log10sigma_v = np.log10(self.energy_transfer_cross_section(10 ** log10v))
        self._energy_transfer_cross_section_interp = interp1d(log10v, log10sigma_v)

    def interpolate_v5_cross_section(self, vmin, vmax, n_steps):

        log10v = np.linspace(np.log10(vmin), np.log10(vmax), n_steps)
        log10sigma_v = np.log10(self.v5_transfer_cross_section(10 ** log10v))
        self._v5_cross_interp = interp1d(log10v, log10sigma_v)

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

    def velocity_weighted_cross_section(self, v_rms, n):
        """
        Averages the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        <sigma(v) v> / <v>
        """

        return self.velocity_weighted_average(v_rms, n) / self.velocity_moment(v_rms, n)

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
            return 10**self._scattering_rate_cross_section_interp(np.log10(v_rms))

    def momentum_transfer_cross_section(self, v_rms):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^2 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        return self.velocity_weighted_average(v_rms, 2)/self.velocity_moment(v_rms, 2)

    def energy_transfer_cross_section(self, v):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^3 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        if self._energy_transfer_cross_section_interp is None:

            if isinstance(v, list) or isinstance(v, np.ndarray):
                integral = [self.velocity_weighted_average(vi, 3)/self.velocity_moment(vi, 3) for vi in v]
                return np.array(integral)
            else:
                return self.velocity_weighted_average(v, 3)/self.velocity_moment(v, 3)

        else:
            return 10**self._energy_transfer_cross_section_interp(np.log10(v))

    def v5_transfer_cross_section(self, v):
        """
        Integrates the velcoity dependence of the cross section over a Maxwell-Boltzmann distribution:

        \int_{0}^{infinity} K(v) * v^5 * sigma(v) dv / <v^2>

        K(v) is the Maxwell Boltzmann kernel: 4 * pi * v^2 * exp(-v^2 / v_p^2) / (pi * v_p^2)^(3/2), where v_p is
        the most probable speed, and v_rms is the velocity dispersion witgh v_p^2 = 2/3 v_rms^2.
        """

        if self._v5_cross_interp is None:

            if isinstance(v, list) or isinstance(v, np.ndarray):
                integral = [self.velocity_weighted_average(vi, 5)/self.velocity_moment(vi, 5) for vi in v]
                return np.array(integral)
            else:
                return self.velocity_weighted_average(v, 5)/self.velocity_moment(v, 5)

        else:
            return 10**self._v5_cross_interp(np.log10(v))

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
        if self.use_trap_z:
            v = np.linspace(self.vmin_integral, self.vmax_integral, 600)
            y = self._integrand(v, *args)
            return np.trapz(y, v)
        else:
            return quad(self._integrand, 0, min(100 * v_rms, 500), args)[0]

    @staticmethod
    def _integrand(v, v_rms, n, func):
        x = v ** 2 / (4 * v_rms ** 2)
        kernel = v ** (2 + n) * np.exp(-x) / (2 * np.sqrt(np.pi) * v_rms ** 3)
        return kernel * func(v)


