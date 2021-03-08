from sidmpy.Profiles.coreTNFW_deflection_angle import CoreTNFWDeflection
import numpy.testing as npt
import numpy as np
from lenstronomy.LensModel.Profiles.tnfw import TNFW

class TestcoreTNFWDeflection(object):

    def setup(self):

        self.interp = CoreTNFWDeflection()
        self.tnfw = TNFW()

    def test_deflection_point(self):

        x = 0.
        y = 0.
        Rs = 0.1
        r_core = 0.04
        r_trunc = 0.5
        norm = 1.

        alpha_origin = self.interp(x, y, Rs, r_core, r_trunc, norm)
        npt.assert_almost_equal(alpha_origin, 0.)

        x = Rs/np.sqrt(2)
        y = Rs/np.sqrt(2)
        alpha_Rs = 10.
        alpha_tnfw,_ = self.tnfw.derivatives(Rs, 0., Rs, alpha_Rs, r_trunc)

        alpha_rs_x, alpha_rs_y = self.interp(x, y, Rs, r_core, r_trunc, norm=1.)
        alpha_rs_interp_x, alpha_rs_interp_y = self.interp(x, y, Rs, r_core,
                                                           r_trunc, norm=alpha_tnfw)

        npt.assert_almost_equal(alpha_rs_interp_x / alpha_rs_x, alpha_tnfw)
        npt.assert_almost_equal(alpha_rs_interp_y / alpha_rs_y, alpha_tnfw)

    def test_deflection_array(self):

        x = np.array([0., 0.])
        y = 0.
        Rs = 0.1
        r_core = 0.04
        r_trunc = 0.5
        norm = 1.

        alpha_origin_x, alpha_origin_y = self.interp(x, y, Rs, r_core, r_trunc, norm)
        npt.assert_almost_equal(alpha_origin_y, 0.)
        npt.assert_almost_equal(alpha_origin_x, 0.)

        x = np.array([Rs / np.sqrt(2), Rs/np.sqrt(2)])
        y = np.array([Rs / np.sqrt(2), Rs/np.sqrt(2)])
        alpha_Rs = 10.
        alpha_tnfw, _ = self.tnfw.derivatives(Rs, 0., Rs, alpha_Rs, r_trunc)

        alpha_rs_interp_x, alpha_rs_interp_y = self.interp(x, y, Rs, r_core, r_trunc, norm=1.)
        alpha_rs = self.interp(x, y, Rs, r_core, r_trunc, norm=alpha_tnfw)
        npt.assert_almost_equal(alpha_rs / alpha_rs_interp_x, alpha_tnfw)
        npt.assert_almost_equal(alpha_rs / alpha_rs_interp_y, alpha_tnfw)

