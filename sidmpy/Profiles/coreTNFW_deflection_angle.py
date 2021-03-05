import numpy as np
from scipy.interpolate import interp1d
from sidmpy.Profiles.deflection_angles_cored_profile import log_deflection_angle
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class CoreTNFWDeflection(object):

    """
    This class interpolates the deflection angles for a cored TNFW parameterized as

    rho(r) ~ 1/ ( (b^a + x^a)^(-1/a) * (1 + x)^2 * (x^2 + t^2) )

    where b = r_c / r_s is a core radius, x = r / r_s, and t = r_t / r_s is a truncation radius.

    The deflection angles themselves have been computed on a grid of b and t, and the shape of each
    deflection angle profile is saved in the file deflection_angles_cored_profile. This class
    reads the file and returns the deflection angle shape that corresponds to the specified values of (b, t). The
    deflection is computed up to a constant numerical prefactor.
    """

    def __init__(self):

        self.deflections = 10 ** log_deflection_angle

        self.tau = np.arange(1, 31, 1)
        self.beta = np.arange(0.0025, 1.005, 0.005)

        log_xnfw = np.log10(np.logspace(-3, 2, 100))

        self.split = []

        self.log_xmin, self.log_xmax = log_xnfw[0], log_xnfw[-1]
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]
        self._delta_beta = self.beta[2] - self.beta[1]
        self._logx_domain = log_xnfw

        self._tau_min = self.tau[0]
        self._tau_max = self.tau[-1]
        self._delta_tau = self.tau[1] - self.tau[0]

        interp_list = []
        for i, bi in enumerate(self.beta):
            interp_list_tau = []
            for j, tj in enumerate(self.tau):

                interp = interp1d(log_xnfw, log_deflection_angle[i, j, :])
                interp_list_tau.append(interp)
            interp_list.append(interp_list_tau)

        self._interp_list = interp_list

        self._tnfw_profile = TNFW()

    def _L(self, x, tau):
        """
        Logarithm that appears frequently
        :param x: r/Rs
        :param tau: t/Rs
        :return:
        """

        return np.log(x * (tau + np.sqrt(tau ** 2 + x ** 2)) ** -1)

    def _F(self, x):
        """
        Classic NFW function in terms of arctanh and arctan
        :param x: r/Rs
        :return:
        """
        if isinstance(x, np.ndarray):
            nfwvals = np.ones_like(x)
            inds1 = np.where(x < 1)
            inds2 = np.where(x > 1)
            nfwvals[inds1] = (1 - x[inds1] ** 2) ** -.5 * np.arctanh((1 - x[inds1] ** 2) ** .5)
            nfwvals[inds2] = (x[inds2] ** 2 - 1) ** -.5 * np.arctan((x[inds2] ** 2 - 1) ** .5)
            return nfwvals

        elif isinstance(x, float) or isinstance(x, int):
            if x == 1:
                return 1
            if x < 1:
                return (1 - x ** 2) ** -.5 * np.arctanh((1 - x ** 2) ** .5)
            else:
                return (x ** 2 - 1) ** -.5 * np.arctan((x ** 2 - 1) ** .5)

    def _tnfw_def(self, x, tau):

        # revert to NFW normalization for now

        factor = tau ** 2 * (tau ** 2 + 1) ** -2 * (
            (tau ** 2 + 1 + 2 * (x ** 2 - 1)) * self._F(x) + tau * np.pi + (tau ** 2 - 1) * np.log(tau) +
            np.sqrt(tau ** 2 + x ** 2) * (-np.pi + self._L(x, tau) * (tau ** 2 - 1) * tau ** -1))

        return factor * x ** -1

    def __call__(self, x, y, Rs, r_core, r_trunc, norm):

        """
        Evaluates then deflection angle (up to a constant numerical factor)
        :param x: x coordinate in arcsec
        :param y: y coordinate in arcsec
        :param Rs: scale radius of profile (arbitrary units)
        :param r_core: core radius of profile (same units as rs)
        :param r_trunc: truncation radius of profile (same units as rs)
        :param norm: a normalization factor such that the return deflection angle equals norm * interp(R/rs)
        :return: The deflection angle at R = np.sqrt(x^2 + y^2)
        """

        beta = r_core/Rs
        tau = r_trunc/Rs

        index_1 = np.argmin(np.absolute(beta - self.beta))
        index_2 = np.argmin(np.absolute(tau - self.tau))

        func = self._interp_list[index_1][index_2]
        R = np.sqrt(x ** 2 + y ** 2)

        if isinstance(R, float) or isinstance(R, int):

            if R == 0:
                return 0., 0.

            xmin = 10 ** self.log_xmin
            R = max(xmin * Rs, R)
            x_nfw = R / Rs
            log_x = np.log10(x_nfw)

            if log_x < self.log_xmin:

                alpha_radial = 0.

            elif log_x < self.log_xmax:

                alpha_radial = 10 ** func(log_x)

            else:

                alpha_interp_at_xmax = 10 ** func(self.log_xmax)
                alpha_nfw_at_xmax = self._tnfw_def(10 ** self.log_xmax, tau)
                rescale = alpha_nfw_at_xmax / alpha_interp_at_xmax
                alpha_radial = rescale * self._tnfw_def(10 ** log_x, tau)

        else:

            xmin = 10 ** self.log_xmin
            R[np.where(R < xmin * Rs)] = xmin * Rs
            x_nfw = R / Rs
            log_x = np.log10(x_nfw)

            alpha_radial = np.zeros_like(R)
            high_inds = np.where(log_x >= self.log_xmax)
            valid_inds = np.where(np.logical_and(log_x > self.log_xmin, log_x < self.log_xmax))[0]

            alpha_radial[valid_inds] = 10 ** func(log_x[valid_inds])

            alpha_interp_at_xmax = 10 ** func(self.log_xmax)
            alpha_nfw_at_xmax = self._tnfw_def(10**self.log_xmax, tau)
            rescale = alpha_nfw_at_xmax/alpha_interp_at_xmax
            alpha_radial[high_inds] = rescale * self._tnfw_def(10**log_x[high_inds], tau)

        alpha_x = norm * alpha_radial * x/R
        alpha_y = norm * alpha_radial * y/R
        return alpha_x, alpha_y
