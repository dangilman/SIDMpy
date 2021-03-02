from sidmpy.CrossSections.velocity_independent import VelocityIndependentCrossSection
import numpy as np
import numpy.testing as npt

class TestCrossSection(object):

    def setup(self):

        self.norm = 5.
        self.cross = VelocityIndependentCrossSection(self.norm)

    def test_scattering_rate(self):

        fac = 4 / np.sqrt(np.pi)
        vrms = 1.
        sigma_v = self.cross.scattering_rate_cross_section(vrms)
        npt.assert_almost_equal(sigma_v, self.norm * fac * vrms)
