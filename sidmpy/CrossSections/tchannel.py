from sidmpy.CrossSections.cross_section import InteractionCrossSection

class TChannel(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section of the form

    d sigma(v)/ domega = 2 * pi * norm / (1 + sin^2(theta/2) * (v/v_ref)^2 )
    """

    def __init__(self, norm, v_ref):

        self._vref = v_ref
        super(TChannel, self).__init__(norm, self._velocity_dependence_kernel)

    def _velocity_dependence_kernel(self, v):

        # 2pi to go from dsigma/domega to total cross section
        two_pi = 6.283185
        r = v / self._vref
        denom = (1 + r ** 2) ** 2
        return two_pi / denom
