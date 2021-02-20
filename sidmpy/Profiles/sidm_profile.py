from sidmpy.sidmpy import solve_from_NFW_params
from sidmpy.Solver.util import integrate_profile
from sidmpy.Profiles.halo_density_profiles import TNFWprofile
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
import numpy as np

class SIDMprofile(object):

    """
    This class smoothly interpolates a cored isothermal profile and a truncated NFW joined at r1
    """

    def __init__(self, rho0, vdis, r1, rhos, rs, xmin=0.01, xmax=50):

        self.r_iso, self.rho_isothermal = integrate_profile(rho0, vdis, rs, r1,
                                              rmin_fac=xmin, rmax_fac=xmax)

        self.rho0 = rho0
        self.r_1 = r1
        self.velocity_dispersion = vdis
        self.rc_over_rs = rhos/rho0
        self.rcore_units_kpc = rs * self.rc_over_rs

        self._rmax = self.r_iso[-1]
        self._rmin = self.r_iso[0]
        r_match_index = np.argmin(np.absolute(self.r_iso - r1))
        self._rmatch = self.r_iso[r_match_index]

        assert self._rmin < self.rcore_units_kpc, 'Must match profiles with some radius < r_core'
        assert self._rmax >= self.r_1, "Must match profiles with rmax > r_1"

        self.rho_iso_interp = interp1d(self.r_iso, self.rho_isothermal)

        self.rhos, self.rs = rhos, rs

    @classmethod
    def fromNFWparams(cls, rhos, rs, halo_age, cross_section_type, cross_section_kwargs,
                      kwargs_solver={}, xmin=0.01, xmax=50):

        """
        Instantiates the class by first solving for the isothermal solution given the parameters describing the NFW
        profile and the SIDM cross section
        :param rhos:
        :param rs:
        :param halo_age:
        :param cross_section_class:
        :param kwargs_solver:
        :param xmin:
        :param xmax:
        :return:
        """
        rho0, vdis, r1 = solve_from_NFW_params(rhos, rs, halo_age, cross_section_type, cross_section_kwargs,
                                               **kwargs_solver)
        return SIDMprofile(rho0, vdis, r1, rhos, rs, xmin, xmax)

    def __call__(self, r, rt=100000, smooth=True, smooth_scale=0.1):

        """
        Evaluates the joined SIDM and NFW density profile as a function of r
        :param r: radius at which to evaluate the profile in kpc
        :param rt: truncation radius of the truncated NFW halo
        :param smooth: if True, convolves the profile with a Gaussian to remove the sharp density profile discontinuity
        at r_1
        :param smooth_scale: sets the smoothing scale length (in units of rs) for the convolution if smooth is True
        :return: the density profile evaluated at r
        """

        if isinstance(r, np.ndarray) or isinstance(r, list):
            r = np.array(r)
            out = np.empty_like(r)
            inds_0 = np.where(r <= self._rmin)
            inds_iso = np.where((r > self._rmin) & (r <= self._rmatch))
            inds_nfw = np.where(r > self._rmatch)
            out[inds_0] = self.rho0
            out[inds_iso] = self.rho_iso_interp(r[inds_iso])
            out[inds_nfw] = TNFWprofile(r[inds_nfw], self.rhos, self.rs, rt)
            if smooth:
                out = gaussian_filter(out, sigma=smooth_scale*self.rs)
            return out

        else:
            if r <= self._rmin:
                return self.rho0
            elif r <= self._rmatch:
                return self.rho_iso_interp(r)
            else:
                return TNFWprofile(r, self.rhos, self.rs, rt)

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
