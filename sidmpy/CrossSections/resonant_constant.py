from sidmpy.CrossSections.cross_section import InteractionCrossSection
from sidmpy.CrossSections.velocity_independent import VelocityIndependentCrossSection


class ResonantConstant(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section of the form

    sigma(v) = norm / (1 + (v/v_ref)^2 )^2

    """

    def __init__(self, norm, v_res, w_res, res_amplitude):

        self._vres = v_res
        self._res_amp = res_amplitude
        self._wres = w_res
        self._vindep = VelocityIndependentCrossSection(norm)
        super(ResonantConstant, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm,
                'v_res': self._vres, 'res_amp': self._res_amp,
                'res_width': self._wres}

    def _velocity_dependence_kernel(self, v):

        kernel_vindep = self._vindep.evaluate(v)/self.norm
        x = (v - self._vres)/self._wres
        kernel_resonance = self._res_amp / (1 + x**2)
        return kernel_resonance/self.norm + kernel_vindep
