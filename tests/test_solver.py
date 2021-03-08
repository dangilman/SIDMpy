from sidmpy.sidmpy import solve_from_NFW_params
from sidmpy.sidmpy import solve_rho_with_interpolation, solve_sigmav_with_interpolation
from sidmpy.Solver.util import nfwprofile_mass, isothermal_profile_density, isothermal_profile_mass, integrate_profile
from sidmpy.Profiles.halo_density_profiles import TNFWprofile
import numpy.testing as npt

class TestSolver(object):

    def setup(self):

        rhos, rs = 34401797.75629296, 0.494554
        halo_age = 8.19
        cross_section_type = 'TCHANNEL'
        kwargs_cross_section = {'norm': 10, 'v_ref': 100}
        self.rho0, self.sigmav, self.r1 = solve_from_NFW_params(rhos, rs, halo_age, cross_section_type, kwargs_cross_section)
        self.rhos = rhos
        self.rs = rs
        self.cross_section_type = cross_section_type
        self.kwargs_cross_section = kwargs_cross_section

    def test_solver(self):

        mass_nfw = nfwprofile_mass(self.rhos, self.rs, self.r1)
        density_nfw = TNFWprofile(self.r1, self.rhos, self.rs, 1000 * self.rs)
        r_iso, rho_iso = integrate_profile(self.rho0, self.sigmav, self.rs, self.r1)
        mass_isothermal = isothermal_profile_mass(r_iso, rho_iso, self.r1)
        density_isothermal = isothermal_profile_density(self.r1, r_iso, rho_iso)
        npt.assert_almost_equal(density_isothermal/density_nfw, 1, 2)
        npt.assert_almost_equal(mass_nfw / mass_isothermal, 1, 2)

    def test_solution_interp(self):

        rho0_interp = solve_rho_with_interpolation(10**8, 0.6, 0., self.cross_section_type,
                                 self.kwargs_cross_section)
        npt.assert_equal(abs(rho0_interp/self.rho0 - 1) < 0.2, True)
        sigmav_interp = solve_sigmav_with_interpolation(10 ** 8, 0.6, 0., self.cross_section_type,
                                                   self.kwargs_cross_section)
        npt.assert_equal(abs(sigmav_interp / self.sigmav - 1) < 0.2, True)

