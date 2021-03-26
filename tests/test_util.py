import numpy as np
import numpy.testing as npt
from sidmpy.Profiles.halo_density_profiles import TNFWprofile
from sidmpy.Solver.util import nfwprofile_mass, nfw_velocity_dispersion_analytic
from scipy.integrate import quad

class TestUtilFunctions(object):

    def test_nfw_velocity_dispersion(self):

        rhos, rs = 24199124.73664613, 0.55074782382099
        rt = 10000 * rs
        r = np.linspace(0.01, 10, 1000) * rs
        vdis_square = np.array([nfw_velocity_dispersion_analytic(ri, rhos, rs) for ri in r]) ** 2
        G = 4.3e-6
        def _integrand(rprime):
            return TNFWprofile(rprime, rhos, rs, rt) * G * nfwprofile_mass(rhos, rs, rprime)/rprime**2

        for i, ri in enumerate(r):
            value = quad(_integrand, ri, 10000 * rs)[0]/TNFWprofile(ri, rhos, rs, rt)
            npt.assert_almost_equal(vdis_square[i]/value, 1, 3)

