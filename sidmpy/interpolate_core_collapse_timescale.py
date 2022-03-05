import numpy as np
from sidmpy.core_collapse_timescale import fraction_collapsed_halos, fraction_collapsed_halos_pool
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import pickle
from multiprocess.pool import Pool

class InterpolatedCollapseTimescale(object):

    def __init__(self, m1, m2, cross_section_model, param_names, param_arrays, params_fixed={},
                 kwargs_fraction={}, step_scale=50, nproc=8):

        self.param_names = param_names
        self.param_ranges = []
        self.param_ranges_dict = {}
        for i, param in enumerate(param_arrays):
            ran = [param[0], param[-1]]
            self.param_ranges.append(ran)
            self.param_ranges_dict[param_names[i]] = ran

        print('param_names: ', param_names)
        print('n params: ', len(param_names))
        print('n sample arrays: ', len(param_arrays))
        # redshift is always last
        if len(param_arrays) == 2:
            args_list = []
            points = (param_arrays[0], param_arrays[1])
            n_total = len(param_arrays[0]) * len(param_arrays[1])
            print('n total: ', n_total)

            for p1 in param_arrays[0]:
                for redshift in param_arrays[1]:
                    kw = {param_names[0]: p1}
                    kw.update(params_fixed)
                    kwargs_fraction['redshift'] = redshift
                    cross_model = cross_section_model(**kw)
                    new = (m1, m2, cross_model, kwargs_fraction['redshift'], kwargs_fraction['timescale_factor'])
                    args_list.append(new)

            shape = (len(param_arrays[0]), len(param_arrays[1]))

        elif len(param_arrays) == 3:

            args_list = []
            points = (param_arrays[0], param_arrays[1], param_arrays[2])
            n_total = len(param_arrays[0]) * len(param_arrays[1]) * len(param_arrays[2])
            print('n total: ', n_total)

            for p1 in param_arrays[0]:
                for p2 in param_arrays[1]:
                    for redshift in param_arrays[2]:
                        # if counter % step == 0:
                        #     print(str(np.round(100 * counter / n_total, 1)) + '% ')
                        kw = {param_names[0]: p1, param_names[1]: p2}
                        kw.update(params_fixed)
                        kwargs_fraction['redshift'] = redshift
                        cross_model = cross_section_model(**kw)
                        new = (m1, m2, cross_model, kwargs_fraction['redshift'], kwargs_fraction['timescale_factor'])
                        args_list.append(new)

            shape = (len(param_arrays[0]), len(param_arrays[1]), len(param_arrays[2]))

        elif len(param_arrays) == 4:

            points = (param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3])
            n_total = len(param_arrays[0]) * len(param_arrays[1]) * len(param_arrays[2]) * len(param_arrays[3])
            print('n total: ', n_total)

            args_list = []

            for p1 in param_arrays[0]:
                for p2 in param_arrays[1]:
                    for p3 in param_arrays[2]:
                        for redshift in param_arrays[3]:

                            kw = {param_names[0]: p1, param_names[1]: p2, param_names[2]: p3}
                            kw.update(params_fixed)
                            kwargs_fraction['redshift'] = redshift
                            cross_model = cross_section_model(**kw)
                            new = (m1, m2, cross_model, kwargs_fraction['redshift'], kwargs_fraction['timescale_factor'])
                            args_list.append(new)


            pool = Pool(nproc)
            values = pool.map(fraction_collapsed_halos_pool, args_list)
            pool.close()
            shape = (len(param_arrays[0]), len(param_arrays[1]), len(param_arrays[2]), len(param_arrays[3]))

        elif len(param_arrays) == 5:

            points = (param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3], param_arrays[4])
            n_total = len(param_arrays[0]) * len(param_arrays[1]) * len(param_arrays[2]) * len(param_arrays[3]) * len(param_arrays[4])
            print('n total: ', n_total)
            args_list = []
            for p1 in param_arrays[0]:
                for p2 in param_arrays[1]:
                    for p3 in param_arrays[2]:
                        for p4 in param_arrays[3]:
                            for redshift in param_arrays[4]:
                                # if counter % step == 0:
                                #     print(str(np.round(100 * counter / n_total, 1)) + '% ')
                                kw = {param_names[0]: p1, param_names[1]: p2, param_names[2]: p3, param_names[3]: p4}
                                kw.update(params_fixed)
                                kwargs_fraction['redshift'] = redshift
                                cross_model = cross_section_model(**kw)
                                new = (
                                m1, m2, cross_model, kwargs_fraction['redshift'], kwargs_fraction['timescale_factor'])
                                args_list.append(new)

        elif len(param_arrays) == 6:

            points = (param_arrays[0], param_arrays[1], param_arrays[2], param_arrays[3], param_arrays[4], param_arrays[5])
            n_total = len(param_arrays[0]) * len(param_arrays[1]) * len(param_arrays[2]) * len(param_arrays[3]) * len(
                param_arrays[4]) * len(param_arrays[5])
            print('n total: ', n_total)
            args_list = []
            for p1 in param_arrays[0]:
                for p2 in param_arrays[1]:
                    for p3 in param_arrays[2]:
                        for p4 in param_arrays[3]:
                            for p5 in param_arrays[4]:
                                for redshift in param_arrays[5]:
                                    # if counter % step == 0:
                                    #     print(str(np.round(100 * counter / n_total, 1)) + '% ')
                                    kw = {param_names[0]: p1, param_names[1]: p2, param_names[2]: p3, param_names[3]: p4,
                                          param_names[4]: p5}
                                    kw.update(params_fixed)
                                    kwargs_fraction['redshift'] = redshift
                                    print(kwargs_fraction)
                                    cross_model = cross_section_model(**kw)
                                    new = (
                                        m1, m2, cross_model, kwargs_fraction['redshift'],
                                        kwargs_fraction['timescale_factor'])
                                    args_list.append(new)

            pool = Pool(nproc)
            values = pool.map(fraction_collapsed_halos_pool, args_list)
            pool.close()
            shape = (len(param_arrays[0]), len(param_arrays[1]), len(param_arrays[2]), len(param_arrays[3]),
                     len(param_arrays[4]))

        else:
            raise Exception('only 2, 3, 4 and 5D interpolations implemented')

        if len(param_names) == 1:
            self._interp_function = interp1d(points, values)
        else:
            self._interp_function = RegularGridInterpolator(points, np.array(values).reshape(shape),
                                                            bounds_error=False, fill_value=None)

    def __call__(self, *args):
        return np.squeeze(self._interp_function(tuple(args)))

def interpolate_collapse_fraction(fname, cross_section_class, param_names, param_arrays, params_fixed, m1,
                                  kwargs_collapse_fraction, nproc):

    interp_timescale = InterpolatedCollapseTimescale(m1, m1 * 1.05, cross_section_class,
                                                     param_names, param_arrays, params_fixed, kwargs_collapse_fraction, nproc=nproc)

    f = open('interpolated_collapse_fraction_'+fname, 'wb')
    pickle.dump(interp_timescale, f)
    f.close()

# from sidmpy.CrossSections.resonant_tchannel import ExpResonantTChannel
# # norm, v_ref, v_res, w_res, res_amplitude
# param_names = ['norm', 'v_ref', 'v_res', 'w_res', 'res_amplitude', 'redshift']
# cross_model = ExpResonantTChannel
#
# output_folder = ''
# nproc = 8
# params_fixed = {}
# kwargs_collapse_fraction = {'timescale_factor': 10.0/3}
# z_array = [0.2, 0.4, 0.6, 0.8, 1.0]
# param_arrays = [np.linspace(1, 10.0, 9), np.linspace(1, 50.0, 20), np.linspace(1, 40, 20),
#                 np.linspace(1, 5.0, 5), np.linspace(1.0, 100, 40), z_array]
# n_total = 1
# for parr in param_arrays:
#     n_total *= len(parr)
# print('n_total: ', n_total); a=input('continue')
# fname = output_folder + 'logM68_expresonanttchannel'
# m1 = 10 ** 7
# interpolate_collapse_fraction(fname, cross_model, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction, nproc=nproc)

# fname = output_folder + 'logM89_expresonanttchannel'
# m1 = 10 ** 8.5
# interpolate_collapse_fraction(fname, cross_model, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction, nproc=nproc)
#
# fname = output_folder + 'logM910_expresonanttchannel'
# m1 = 10 ** 9.5
# interpolate_collapse_fraction(fname, cross_model, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction, nproc=nproc)


# from sidmpy.CrossSections.tchannel import TChannel
# param_names = ['norm', 'v_ref']
# n = 50
# params_fixed = {}
# kwargs_collapse_fraction = {'redshift': 0.5, 'timescale_factor': 20.0}
# param_arrays = [np.linspace(0.5, 60.0, n), np.linspace(1.0, 40, n)]
# fname = 'logM68_tchannel'
# m1 = 10 ** 7
# interpolate_collapse_fraction(fname, TChannel, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction)
#
# fname = 'logM89_tchannel'
# m1 = 5 * 10 ** 8
# interpolate_collapse_fraction(fname, TChannel, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction)
#
# fname = 'logM910_tchannel'
# m1 = 5 * 10 ** 9
# interpolate_collapse_fraction(fname, TChannel, param_names, param_arrays, params_fixed, m1, kwargs_collapse_fraction)
#
