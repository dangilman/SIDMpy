import numpy as np
from sidmpy.CrossSections.cross_section import InteractionCrossSection
from scipy.interpolate import interp1d
import inspect
local_path = inspect.getfile(inspect.currentframe())[0:-20]

_log10velocity0 = np.log10(np.logspace(0, np.log10(300), 101, endpoint=True))
_fname = local_path+'/benchmark0_cross.txt'
_log10cross0 = np.loadtxt(_fname)
_cross_interp0 = interp1d(_log10velocity0, _log10cross0)

_log10velocity1 = np.log10(np.logspace(0, np.log10(300), 51, endpoint=True))
_fname = local_path+'/benchmark1_cross.txt'
_log10cross1 = np.loadtxt(_fname)
_cross_interp1 = interp1d(_log10velocity1, _log10cross1)

_log10velocity2 = np.log10(np.logspace(0, np.log10(300), 301, endpoint=True))
_fname = local_path+'/benchmark2_cross.txt'
_log10cross2 = np.loadtxt(_fname)
_cross_interp2 = interp1d(_log10velocity2, _log10cross2)

_log10velocity3 = np.log10(np.append(np.logspace(-1, -0.1, 10),np.append(np.append(np.append(np.logspace(0, np.log10(7), 31, endpoint=True),
                            np.logspace(np.log10(7.25), np.log10(12.), 91, endpoint=True)),
                            np.logspace(np.log10(12.5), np.log10(55.), 41, endpoint=True)),
                            np.logspace(np.log10(60.), np.log10(300.), 16, endpoint=True))))
_fname = local_path+'/benchmark3_cross.txt'
_log10cross3 = np.loadtxt(_fname)
_cross_interp3 = interp1d(_log10velocity3, _log10cross3)

_log10velocity4 = np.log10(np.logspace(0, np.log10(300), 51, endpoint=True))
_fname = local_path+'/benchmark4_cross.txt'
_log10cross4 = np.loadtxt(_fname)
_cross_interp4 = interp1d(_log10velocity4, _log10cross4)

_log10velocity5 = np.log10(np.logspace(0, np.log10(300), 101, endpoint=True))
_fname = local_path+'/benchmark5_cross.txt'
_log10cross5 = np.loadtxt(_fname)
_cross_interp5 = interp1d(_log10velocity5, _log10cross5)

_log10velocity6 = np.log10(np.logspace(0, np.log10(400), 81, endpoint=True))
_fname = local_path+'/benchmark6_cross.txt'
_log10cross6 = np.loadtxt(_fname)
_cross_interp6 = interp1d(_log10velocity6, _log10cross6)

_log10velocity7 = np.log10(np.logspace(0, np.log10(400), 101, endpoint=True))
_fname = local_path+'/benchmark7_cross.txt'
_log10cross7 = np.loadtxt(_fname)
_cross_interp7 = interp1d(_log10velocity7, _log10cross7)

_log10velocity8 = np.log10(np.logspace(0, np.log10(300), 201, endpoint=True))
_fname = local_path+'/benchmark8_cross.txt'
_log10cross8 = np.loadtxt(_fname)
_cross_interp8 = interp1d(_log10velocity8, _log10cross8)

_log10velocity9 = np.log10(np.logspace(0, np.log10(300), 81, endpoint=True))
_fname = local_path+'/benchmark9_cross.txt'
_log10cross9 = np.loadtxt(_fname)
_cross_interp9 = interp1d(_log10velocity9, _log10cross9)

_log10velocity10 = np.log10(np.logspace(-1., np.log10(300), 51, endpoint=True))
_fname = local_path+'/benchmark10_cross.txt'
_log10cross10 = np.loadtxt(_fname)
_cross_interp10 = interp1d(_log10velocity10, _log10cross10)

class Benchmark0(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp0
        norm = 1

        super(Benchmark0, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark0(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark0(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.4,
                'alpha_chi': -0.00091}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark1(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp1
        norm = 1

        super(Benchmark1, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark1(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark1(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -3.75,
                'alpha_chi': -0.000563}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark2(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp2
        norm = 1

        super(Benchmark2, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark2(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark2(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -3.75,
                'alpha_chi': -0.0015848}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300/v)**4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark3(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp3
        norm = 1

        super(Benchmark3, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark3(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark3(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.549,
                'alpha_chi': -0.00154881}

    def _extrapolate(self, v):

        if v < 0.1:
            log10v = -1.
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark4(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp4
        norm = 1

        super(Benchmark4, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark4(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark4(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.,
                'alpha_chi': -0.00059}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark5(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp5
        norm = 1

        super(Benchmark5, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark5(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark5(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.1,
                'alpha_chi': -0.00095499}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark6(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp6
        norm = 1

        super(Benchmark6, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark6(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark6(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.4,
                'alpha_chi': -0.00091}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 400:
            log10v = np.log10(400)
            match = 10 ** self._interp(log10v)
            return match * (400 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark7(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp7
        norm = 1

        super(Benchmark7, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark7(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark7(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -3.5,
                'alpha_chi': -0.009}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 400:
            log10v = np.log10(400)
            match = 10 ** self._interp(log10v)
            return match * (400 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark8(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp8
        norm = 1

        super(Benchmark8, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark8(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark8(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.5,
                'alpha_chi': -0.0033}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark9(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp9
        norm = 1

        super(Benchmark9, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark9(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark9(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -4.02,
                'alpha_chi': -10**-2.89}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 1.0:
            log10v = 0.0
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3

class Benchmark10(InteractionCrossSection):

    def __init__(self, m_chi, auto_interp=True):

        self._vmin_integration = 0.1
        self._vmax_integration = 10**3.0
        self._mchi = m_chi
        self._interp = _cross_interp10
        norm = 1

        super(Benchmark10, self).__init__(norm, self._velocity_dependence_kernel,
                                          self._vmin_integration, self._vmax_integration,
                                          use_trap_z=True, auto_interp=auto_interp)

    @classmethod
    def scaled(cls, amplitude_at_v, v, auto_interp=True):

        _cross = Benchmark10(1.0, auto_interp=False)
        amp = _cross.evaluate(v)
        rescale = amplitude_at_v/amp
        m_chi = rescale**(-1/3)
        return Benchmark10(m_chi, auto_interp)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'m_chi': self._mchi,
                'm_phi': self._mchi * 10 ** -5.5,
                'alpha_chi': 0.003}

    @property
    def kappa_beta(self, v=30):
        """
        :param v: velocity in km/sec
        returns the values of kappa, beta, a, and b that delineate different regimes (e.g. quantum vs. semi-classical)
        """
        c = 299729
        voverc = v / c
        kw = self.kwargs
        mr = kw['m_phi'] / kw['m_chi']
        beta = 2 * kw['alpha_chi'] * mr / voverc ** 2
        k = kw['m_chi'] * voverc / 2
        kappa = k / kw['m_phi']
        a = voverc / (2 * kw['alpha_chi'])
        b = kw['alpha_chi'] / mr
        return {'kappa': kappa,
                'beta': beta,
                'twobetak2': 2 * beta * kappa ** 2,
                'a': a,
                'b': b}

    def _extrapolate(self, v):

        if v < 0.1:
            log10v = np.log10(0.1)
            return 10 ** self._interp(log10v)
        elif v > 300:
            log10v = np.log10(300)
            match = 10 ** self._interp(log10v)
            return match * (300 / v) ** 4
        else:
            log10v = np.log10(v)
            return 10 ** self._interp(log10v)

    def _velocity_dependence_kernel(self, v):

        if isinstance(v, float) or isinstance(v, int):
            return self._extrapolate(v)/self._mchi**3
        else:
            out = [self._extrapolate(vi) for vi in v]
            return np.array(out)/self._mchi**3
