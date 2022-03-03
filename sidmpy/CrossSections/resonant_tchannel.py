from sidmpy.CrossSections.cross_section import InteractionCrossSection
from sidmpy.CrossSections.tchannel import TChannel
import numpy as np


class ResonantTChannel(InteractionCrossSection):

    """
    This implements a velocity-dependent cross section of the form

    sigma(v) = norm / (1 + (v/v_ref)^2 )^2 + s / (1 + x^2)
    where x = (v - v_res) / w_res describes resonant feature

    """

    def __init__(self, norm, v_ref, v_res, w_res, res_amplitude):

        self._vref = v_ref
        self._vres = v_res
        self._res_amp = res_amplitude
        self._wres = w_res
        self._tchannel = TChannel(norm, v_ref)
        super(ResonantTChannel, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'v_ref': self._vref,
                'v_res': self._vres, 'res_amp': self._res_amp,
                'res_width': self._wres}

    def _velocity_dependence_kernel(self, v):

        kernel_tchannel = self._tchannel.evaluate(v)/self.norm
        x = (v - self._vres)/self._wres
        kernel_resonance = self._res_amp / (1 + x**2)
        return kernel_resonance/self.norm + kernel_tchannel

class ExpResonantTChannel(ResonantTChannel):

    """
    This implements a velocity-dependent cross section of the form

    sigma(v) = norm / (1 + (v/v_ref)^2 )^2 + s * exp(-x^2/2)
    where x = (v - v_res) / w_res describes resonant feature

    """

    def _velocity_dependence_kernel(self, v):

        kernel_tchannel = self._tchannel.evaluate(v)/self.norm
        x = (v - self._vres)/self._wres
        kernel_resonance = self._res_amp * np.exp(-0.5 * x**2)
        return kernel_resonance/self.norm + kernel_tchannel

