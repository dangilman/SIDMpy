import numpy as np
from scipy.interpolate import interp1d

message = 'could not import deflection angles. If you want to use the interpolated deflection angles ' \
          'for an SIDM profile stored in cnfwmodtrunc_deflections.zip you first have to unzip the file, and then ' \
          'run setup.py develop --user so that the file containing deflection angles will be added to ' \
          'the local python path.'
try:
    from sidmpy.Profiles.corenfw_deflections import deflections
except:
    print('WARNING: '+message)
"""
When using this class, must first unzip the file cnfwmodtrunc_deflections.py.zip to access the
deflection angles stored there. So that they can imported, you should install this package AFTER
unzipping the file.'
"""

class CoreTNFWDeflection(object):

    def __init__(self):

        try:
            self.deflections = deflections
        except:
            print('ERROR: ' + message)
            exit(1)

        self.beta = np.arange(0.01, 1.11, 0.01)
        self.tau = np.linspace(1, 35, 35)

        log_xnfw = np.linspace(-4, 4, 3501)[0:2626]

        self.split = []

        self._xmin, self._xmax = log_xnfw[0], log_xnfw[-1]
        self._betamin = self.beta[0]
        self._betamax = self.beta[-1]
        self._delta_beta = self.beta[2] - self.beta[1]

        self._tau_min = self.tau[0]
        self._tau_max = self.tau[-1]
        self._delta_tau = self.tau[1] - self.tau[0]

        for i, ti in enumerate(self.tau):

            tau_split = []

            for k, bk in enumerate(self.beta):

                tau_split.append(interp1d(log_xnfw, self.deflections[:,i,k]))

            self.split.append(tau_split)

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

    def _get_closest_tau(self, tau_value, ind):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[ind]

        return minidx

    def _get_closest_tau_double(self, tau_value):

        minidx = np.argsort(np.absolute(tau_value - self.tau))[0:2]

        return minidx

    def _get_closest_beta_double(self, beta_value):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[0:2]

        return minidx

    def _get_closest_beta(self, beta_value, ind):

        minidx = np.argsort(np.absolute(beta_value - self.beta))[ind]

        return minidx

    def _euclidean(self, xref, yref, xval, yval):

        return ((xref - xval)**2 + (yref - yval)**2)**0.5

    def __call__(self, x, y, rs, r_core, r_trunc, norm):

        R = np.sqrt(x ** 2 + y ** 2)
        log_xnfw = np.log10(R/rs)

        beta = r_core/rs
        tau = r_trunc/rs

        tmin = self._get_closest_tau(tau, 0)
        bmin = self._get_closest_beta_double(beta)

        if isinstance(R, float) or isinstance(R, int):

            if log_xnfw <= self._xmin:

                defl = self.split[tmin][bmin[0]](self._xmin)
                return norm * defl

            elif log_xnfw > self._xmax:

                defl = self._tnfw_def(10**log_xnfw, tau)
                defl_interp = norm * self.split[tmin][bmin[0]](self._xmax)
                defl_norm = self._tnfw_def(10**self._xmax, tau)

                return defl * (defl_interp / defl_norm)

            else:

                return norm * self.split[tmin][bmin[0]](log_xnfw)

        else:
            #eps = 0

            log_xnfw[np.where(log_xnfw<self._xmin)] = self._xmin
            high_inds = np.where(log_xnfw > self._xmax)
            valid_range = np.where(log_xnfw <= self._xmax)

        alpha = np.empty_like(R)

        alpha[valid_range] = norm * self.split[tmin][bmin[0]](log_xnfw[valid_range])

        defl_interp = norm * self.split[tmin][bmin[0]](self._xmax)
        defl_norm = self._tnfw_def(10 ** self._xmax, tau)

        alpha[high_inds] = self._tnfw_def(10**log_xnfw[high_inds], tau) * (defl_interp / defl_norm)

        return deflections

