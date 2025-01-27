from sidmpy.CrossSections.velocity_independent import VelocityIndependentCrossSection
from sidmpy.CrossSections.tchannel import TChannel
import numpy as np
import numpy.testing as npt

class TestCrossSection(object):

    def setup(self):

        self.norm = 1.
        self.cross_vindep = VelocityIndependentCrossSection(self.norm)
        self.cross_tchannel_vindep = TChannel(self.norm, 1000)

    def test_scattering_rate(self):

        fac = 4 / np.sqrt(np.pi)
        vrms = 1.
        sigma_v = self.cross_vindep.scattering_rate_cross_section(vrms)
        npt.assert_almost_equal(sigma_v, self.norm * fac * vrms)
        sigma_v_tchannel = self.cross_tchannel_vindep.scattering_rate_cross_section(vrms)
        npt.assert_almost_equal(sigma_v_tchannel, self.norm * fac, vrms)

    def test_effective_crosssection(self):

        sigma_effective = self.cross_vindep.effective_cross_section(10)
        npt.assert_almost_equal(sigma_effective, 1)

t = TestCrossSection()
t.setup()
t.test_effective_crosssection()
