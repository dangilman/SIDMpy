from sidmpy.Solver.solve_isothermal import _solve_iterative
from sidmpy.Solver.util import integrate_profile
from sidmpy.Solver.util import compute_r1
from sidmpy.Profiles.nfw import *
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from scipy.integrate import quad
import numpy as np
from sidmpy.CrossSections.power_law import PowerLaw

class SIDMprofile(object):

    def __init__(self, M, c, z, cross_section_norm, v_power, N_solve=5, plot=False,
                 rmax_fac=10, rmin_fac=0.001, x_min=None, x_max=None, astropy=None,
                 z_collapse=10):

        if astropy is None:
            H0 = 69.7
            Ob0 = 0.0464
            Om0 = 0.235
            kwargs_cosmo = {'H0': H0, 'Ob0': Ob0, 'Om0': Om0}
            astropy = FlatLambdaCDM(**kwargs_cosmo)

        thalo = astropy.age(z).value - astropy.age(z_collapse).value
        if thalo < 0:
            raise Exception('You have specified collapse redshift and halo redshift such that the halo has negative age.')

        # NFW parameters in units solar mass and kpc
        rhonfw, rs_nfw = NFW_params_physical(M, c, z, astropy)

        cross_section = PowerLaw(cross_section_norm, v_dep=v_power)

        rho0, s0, core_size_unitsrs, fit_quality, keywords = \
            _solve_iterative(rhonfw, rs_nfw, cross_section, thalo, rmin_fac, rmax_fac,
        N_solve, plot=plot)

        r_1 = compute_r1(rhonfw, rs_nfw, s0, cross_section, thalo)

        if x_max is None or x_max is None:
            r, rho_isothermal = integrate_profile(rho0, s0, rs_nfw, r_1,
                                              rmin_fac=rmin_fac, rmax_fac=rmax_fac)
        else:

            r_min, r_max = x_min * rs_nfw, x_max * rs_nfw

            r, rho_isothermal = integrate_profile(rho0, s0, rs_nfw, r_1,
                                              r_min=r_min, r_max=r_max)

        self.keywords_profile = keywords

        self.rho0 = rho0
        self.r_1 = r_1
        self.velocity_dispersion = s0
        self.rcore_units_kpc = rs_nfw * core_size_unitsrs
        self.rc_over_rs = core_size_unitsrs

        self.r_iso = r
        self.rho_isothermal = rho_isothermal
        self.rmax = r[-1]
        self.rmin = r[0]
        self._rmatch = core_size_unitsrs * rs_nfw

        assert self.rmin < self._rmatch

        if self.rmax <= self.r_1:
            print(self.rmax)
            print(self.r_1)
            raise Exception('must choose a larger value of rmax_fac')

        self.rho_iso_interp = interp1d(r, rho_isothermal)

        self.rhos, self.rs = rhonfw, rs_nfw

        self._rmax_mass = 10000*rs_nfw
        r_values = np.linspace(self.rmin, self._rmax_mass, 500)
        M = [self.enclosed_mass(ri) for ri in r_values]
        self._mass_enclosed_interp = interp1d(r_values, M, fill_value=0)

    def sample_from_profile(self, Npoints, rmin, rmax):

        points = []
        rvalues = np.linspace(rmin, rmax, 20000)
        profile = self(rvalues)
        profile_normed = profile/np.max(profile)
        interp = interp1d(rvalues, profile_normed)

        while True:

            r = np.random.uniform(rmin, rmax)
            u = interp(r)[0]
            p = np.random.rand()
            if u>p:
                points.append(r)
                print(len(points))
                if len(points) > Npoints:
                    break
                if len(points)/10 == 0:
                    percent_done =  np.round(len(points)/Npoints, 1)
                    print(str(percent_done)+'%...')

        return np.array(points)

    def enclosed_mass(self, r):

        if hasattr(self, '_mass_enclosed_interp'):
            return self._mass_enclosed_interp(r)

        M = quad(self._mass_enclosed_integrand, self.rmin, r)[0]
        return M

    def radial_velocity_dispersion(self, r):

        rmax = self._rmax_mass
        (rho, _) = self(r)
        integral = quad(self._velocity_dispersion_integrand, r, rmax)[0]
        return np.sqrt(integral / rho)

    def _velocity_dispersion_integrand(self, r):

        (rho, _) = self(r)
        return rho * self.enclosed_mass(r) / r ** 2

    def _mass_enclosed_integrand(self, r):
        (rho, _) = self(r)
        return 4 * np.pi * r ** 2 * rho

    def __call__(self, r, rt=100000, smooth=False, smooth_scale=0.01):


        if isinstance(r, np.ndarray) or isinstance(r, list):
            out = np.empty_like(r)

            inds_0 = np.where(r <= self.rmin)
            inds_iso = np.where((r > self.rmin) & (r <= self._rmatch))
            inds_nfw = np.where(r > self._rmatch)

            out[inds_0] = 0.
            out[inds_iso] = self.rho_iso_interp(r[inds_iso])
            out[inds_nfw] = TNFWprofile(r[inds_nfw], self.rhos, self.rs, rt)

            if smooth:
                out = gaussian_filter(out, sigma=smooth_scale*self.rs)
            return out, TNFWprofile(r, self.rhos, self.rs, rt)

        else:

            if r <= self.rmin:
                return 0., 0
            elif r <= self._rmatch:
                return self.rho_iso_interp(r), TNFWprofile(r, self.rhos, self.rs, rt)
            else:
                return TNFWprofile(r, self.rhos, self.rs, rt), TNFWprofile(r, self.rhos, self.rs, rt)
