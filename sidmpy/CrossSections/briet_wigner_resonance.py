from sidmpy.CrossSections.cross_section import InteractionCrossSection
from sidmpy.CrossSections.velocity_independent import VelocityIndependentCrossSection
from math import e


class BreitWigner(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section with a resonance at a particular velocity scale v_ref

    the cross section is given as a sum of a velocity-independent piece with amplitude sigma_0, plus a resonance term
    given by the Breit-Wigner resonance function. The full form of the cross section is

    sigma(v) = norm * ( 1 +  k * ((v - v_ref)^2 + w^2)^-1)

    The height of the resonance is given by k, the position of the resonance is determined by v_ref, and the width
    of the resonance is determined by w.
    """

    def __init__(self, norm, v_ref, k, w):

        """
        Note that the normaliztion convention is that the entire cross section is multiplied by "norm"
        in the InteractionCrossSection base class. See the evaluate method in InteractionCrossSection
        :param norm: the overall normalization of the cross section
        (for this model, norm is the cross section amplitude far from the resonance; it can be arbitrarily small,
        but must be non-zero)
        :param v_ref: the position of the resonance
        :param k: the height of the resonance
        :param w: the width of the resonance
        """
        self._vref = v_ref
        self._k = k
        self._w = w
        self._norm = norm
        self._v_indep_cross = VelocityIndependentCrossSection(norm)

        super(BreitWigner, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'v_ref': self._vref,
                'k': self._k, 'w': self._w}

    def _velocity_dependence_kernel(self, v):

        return  (1 - e**(-((v-self._vref)/self._w)**2)) + (self._k/self._norm) * self._w**2 * ((v-self._vref)**2 + self._w**2)**(-1)
