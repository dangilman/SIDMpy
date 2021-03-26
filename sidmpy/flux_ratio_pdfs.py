from lenstronomywrapper.LensSystem.quad_lens import QuadLensSystem
from lenstronomywrapper.LensSystem.macrolensmodel import MacroLensModel
from lenstronomywrapper.LensSystem.LensComponents.powerlawshear import PowerLawShear
from lenstronomywrapper.LensData.lensed_quasar import LensedQuasar
from lenstronomywrapper.Optimization.quad_optimization.hierarchical import HierarchicalOptimization
import matplotlib.pyplot as plt

from pyHalo.preset_models import CDM
from pyHalo.realization_extensions import RealizationExtensions

import numpy as np

from lenstronomy.Util.param_util import phi_q2_ellipticity

from sidmpy.Profiles.coreTNFW_deflection_angle import CoreTNFWDeflection
from sidmpy.sidmpy import solve_rho_with_interpolation
from sidmpy.CrossSections.tchannel import TChannel
from sidmpy.core_collapse_timescale import evolution_timescale_scattering_rate
from sidmpy.sidmpy import solve_sigmav_with_interpolation


def realization_from_args(c0, v0, zlens, zsource, sigma_sub=0.03, LOS_norm=1., cdm=False):
    kwargs_rendering = {'cone_opening_angle': 6, 'log_m_host': 13.3}

    if cdm:
        kwargs_rendering['mass_definition'] = 'TNFW'
    else:
        kwargs_rendering['mass_definition'] = 'coreTNFW'
        deflection_angle_computation = CoreTNFWDeflection()

        kwargs_cross_section = {'norm': c0, 'v_ref': v0}
        kwargs_sidm = {'cross_section_type': 'TCHANNEL', 'kwargs_cross_section': kwargs_cross_section,
                       'SIDM_rhocentral_function': solve_rho_with_interpolation,
                       'numerical_deflection_angle_class': deflection_angle_computation}
        kwargs_rendering.update(kwargs_sidm)

    realization = CDM(zlens, zsource, LOS_normalization=LOS_norm, sigma_sub=sigma_sub, **kwargs_rendering)
    if not cdm:
        ext = RealizationExtensions(realization)
        cross_section = TChannel(**kwargs_cross_section)
        inds = ext.find_core_collapsed_halos(evolution_timescale_scattering_rate, solve_sigmav_with_interpolation,
                                         cross_section, t_sub=10, t_field=100)

        #print('fraction of halos core collapsed: ', len(inds) / len(realization.halos))
        realization = ext.add_core_collapsed_halos(inds, log_slope_halo=3.,
                                               x_core_halo=0.025)
    return realization


def flux_ratios_modeled(args):
    (c0, v0, sigma_sub, LOS_norm, ximg, yimg, magimg, zlens, zsource, macromodel,
     source_size, verbose, cdm) = args
    np.random.seed()
    real = realization_from_args(c0, v0, zlens, zsource, LOS_norm=LOS_norm, sigma_sub=sigma_sub, cdm=cdm)
    data_to_fit = LensedQuasar(ximg, yimg, magimg)
    lens_system = QuadLensSystem.shift_background_auto(data_to_fit, macromodel, zsource, real)
    optimizer = HierarchicalOptimization(lens_system, settings_class='default_CDM2')
    kwargs, lensmodel, return_kwargs = optimizer.optimize(data_to_fit,
                                                                           param_class_name='free_shear_powerlaw',
                                                                           constrain_params=None, verbose=verbose,
                                                                           check_bad_fit=True)
    if lensmodel is None:
        return np.array([np.nan] * 3)
    flux_ratios = lens_system.quasar_magnification(ximg,
                                                   yimg, source_size, lens_model=lensmodel,
                                                   kwargs_lensmodel=kwargs, grid_axis_ratio=0.5,
                                                   grid_resolution_rescale=2., source_model='GAUSSIAN')
    return flux_ratios[1:]/flux_ratios[0]


def forward_model_fluxes(Nreal, c0, v0, source_size, sigma_sub, macro_lens_system, source_x, source_y,
                         verbose=False, pool=None, LOS_norm=1., cdm=False):

    # Compute the image positions using a fast ray-tracing algorithm

    macro_lens_system.update_source_centroid(source_x, source_y)
    lensmodel_init, kwargs_lensmodel_init = macro_lens_system.get_lensmodel()
    x_image, y_image = macro_lens_system.solve_lens_equation(lensmodel_init, kwargs_lensmodel_init)
    magimg = macro_lens_system.quasar_magnification(x_image, y_image, source_size, lensmodel_init,
                                                    kwargs_lensmodel_init, normed=True)

    zlens, zsource = macro_lens_system.zlens, macro_lens_system.zsource
    macromodel = macro_lens_system.macromodel

    shape = (int(Nreal), 3)
    fr_array = np.empty(shape)

    if pool is not None:
        args_list = [(c0, v0, sigma_sub, LOS_norm,
                x_image, y_image, magimg, zlens, zsource,
                macromodel, source_size, verbose, cdm)] * Nreal
        out = pool.map(flux_ratios_modeled, args_list)
        for i in range(0, len(out)):
            fr_array[i, :] = out[i]
    else:
        for i in range(0, int(Nreal)):
            args = (c0, v0, sigma_sub, LOS_norm,
                    x_image, y_image, magimg, zlens, zsource,
                    macromodel, source_size, verbose, cdm)

            fr_array[i, :] = flux_ratios_modeled(args)
            fr_array[i, :] *= fr_array[i, 0] ** -1

    ratios = magimg[1:]/magimg[0]
    return fr_array, ratios
