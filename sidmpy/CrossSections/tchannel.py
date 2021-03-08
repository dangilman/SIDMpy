from sidmpy.CrossSections.cross_section import InteractionCrossSection

class TChannel(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section of the form

    sigma(v) = norm / (1 + (v/v_ref)^2 )^2

    """

    def __init__(self, norm, v_ref):

        self._vref = v_ref
        super(TChannel, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'v_ref': self._vref}

    def _velocity_dependence_kernel(self, v):

        r = v / self._vref
        denom = (1 + r ** 2) ** 2
        return 1 / denom
