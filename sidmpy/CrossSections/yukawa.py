from sidmpy.CrossSections.cross_section import InteractionCrossSection
import numpy as np


class AttractiveYukawa(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section of the form:

    sigma_max * (4 pi / 22.7) * b^2 * log(1 + 1/b) for b < 0.1
    sigma_max * (8 pi / 22.7) * b^2 * (1 + 1.5 * b^1.65)^-1 for 0.1 < b < 10^3
    sigma_max * (pi / 22.7) * [log(b) + 1 - 1/(2 * log(b)) ]^2 for b > 10^3

    where b = pi * (v_max / v)^2, and sigma_max is the amplitude of <sigma(v) v> at v_max

    """

    def __init__(self, norm, v_ref):

        self._vmax = v_ref

        super(AttractiveYukawa, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'v_ref': self._vmax}

    def _beta(self, v):

        return np.pi * (self._vmax / v)**2

    def _vdep_kernel_single_point(self, b):

        pi = np.pi
        coef = pi / 22.7

        if isinstance(b, float) or isinstance(b, int):
            if b <= 0.1:
                return 3.23 * coef * b ** 2 * np.log(1 + 1 / b)
            elif b <= 10 ** 3:
                return 8 * coef * b ** 2 * (1 + 1.5 * b ** 1.65) ** -1
            else:
                return 0.975 * coef * (np.log(b) + 1 - 1 / (2 * np.log(b))) ** 2


    def _velocity_dependence_kernel(self, v):

        b = self._beta(v)

        if isinstance(v, float) or isinstance(v, int):

            return self._vdep_kernel_single_point(b)

        else:

            b = np.array(b)
            shape0 = b.shape
            b = b.ravel()
            out = [self._vdep_kernel_single_point(bi) for bi in b]
            out = np.array(out).reshape(shape0)

            return out

