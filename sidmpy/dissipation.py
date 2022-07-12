import numpy as np
import inspect
import pickle

local_path = inspect.getfile(inspect.currentframe())[0:-15]
f = open(local_path+'/dissipation_factor_interpolation', 'rb')
_dissipation_interp = pickle.load(f)
f.close()

def _compute_grid():
    dissipative_data = np.loadtxt('3d_xi_dissipative_SIDM.dat')
    log10sigma = dissipative_data[:, 0]
    log10nuloss = dissipative_data[:, 1]
    log10xi = dissipative_data[:, 2]

    sigma_values = np.unique(log10sigma)
    nuloss_values = np.unique(log10nuloss)

    values = []
    for sigma in sigma_values:
        for nu in nuloss_values:
            cond1 = sigma == log10sigma
            cond2 = nu == log10nuloss
            idx = np.where(np.logical_and(cond1, cond2))[0][0]
            values.append(log10xi[idx])

    print(min(sigma_values), max(sigma_values))
    print(min(nuloss_values), max(nuloss_values))
    values = np.array(values).reshape(20, 71)
    from scipy.interpolate import RegularGridInterpolator
    points = (sigma_values, nuloss_values)
    grid = RegularGridInterpolator(points, values)

    import pickle
    f = open('dissipation_factor_interpolation', 'wb')
    pickle.dump(grid, f)
    f.close()

    sigma_points = np.random.uniform(min(sigma_values), max(sigma_values), 5000000)
    nu_points = np.random.uniform(min(nuloss_values), max(nuloss_values), 5000000)
    x = np.column_stack((sigma_points, nu_points))

    z = grid(x)
    print(z)
    import matplotlib.pyplot as plt
    h, _, _ = np.histogram2d(nu_points, sigma_points, weights=z, bins=100)
    plt.imshow(h, origin='lower', cmap='seismic')
    plt.show()

def dissipation_timescale_impact(v, cross_section, log10nu_loss, f_dissipative):

    # sigma = cross_section.energy_transfer_cross_section(v)
    # sigma_prime = f_dissipative * sigma
    log10sigma_prime_over_sigma = np.log10(f_dissipative)
    log_xi = _dissipation_interp((log10sigma_prime_over_sigma, log10nu_loss))
    return 10**log_xi
