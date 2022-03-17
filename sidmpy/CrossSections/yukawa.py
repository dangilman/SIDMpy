from sidmpy.CrossSections.cross_section import InteractionCrossSection


class SWaveResonance(InteractionCrossSection):

    """
    A cross section model that scales v^-2 at low v and v^-4 at high v
    The cross section is regularized at v << 1 km/sec to avoid infintities

    norm specifies the amplitude at 30 km/sec
    """
    def __init__(self, norm, vref, low_v_exponent, v_match=30):

        self.v_match = v_match
        self.norm = norm
        self.vref = vref
        self.low_v_exponent = low_v_exponent
        super(SWaveResonance, self).__init__(self.norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'verf': self.vref, 'low_v_exponent': self.low_v_exponent}

    def _velocity_dependence_kernel(self, v):

        v_regularize = 1

        high_v_exponent = 2 - self.low_v_exponent

        amp_at_vmatch = (1 + (self.v_match/v_regularize) ** self.low_v_exponent) * (1 + (self.v_match/self.vref)**2) ** high_v_exponent

        denom = (1 + (v/v_regularize) ** self.low_v_exponent) * (1 + (v/self.vref)**2) ** high_v_exponent

        return amp_at_vmatch/denom

