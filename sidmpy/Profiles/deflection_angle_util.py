import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

def r3d(r2d, z):
    """
    Three dimensional density as a function of z and two-d density r2d
    """
    return np.sqrt(r2d ** 2 + z ** 2)

def integrand_mproj(z, r2d, rhofunc, args):
    """
    The integrand for the projection integral
    sigma(R) = \int_{-\infty}^{\infty} \rho\left(r\left(R, z\right)\right) dz
    """
    try:
        return 2 * rhofunc(r3d(r2d, z), *args)
    except:
        return 2 * rhofunc(r3d(r2d, z), args)

def integrand_deflection(r, rhofunc, args):
    """
    The integrand for the deflection integral
    deflection(R) \sim \frac{2}{R} \int_0^{R} r * sigma(r) dr
    """
    return r * projected_mass(r, rhofunc, args)

def projected_mass(R2D, rho_function, function_args):
    """
    Computes the projection integral
    :param R2D:
    :param rho_function:
    :param function_args:
    :return:
    """
    return quad(integrand_mproj, 0, 500, args=(R2D, rho_function, function_args))[0]

def deflection_point(args):
    """
    Computes the deflection angle at R
    :param R2D:
    :param rho_function:
    :param function_args:
    :return:
    """
    (R, rho_function, function_args) = args
    return (2 / R) * quad(integrand_deflection, 0, R, args=(rho_function, function_args))[0]

def deflection(Rvalues, rho_function, function_args,
               use_pool=False, nproc=10):

    """

    :param Rvalues: r coordinates in 3d
    :param rho_function: a function that outputs the 3d density given a 3d r coordinate

    Must be of the form

    def rho_function(r3d, arg1, arg2, ...):
        return density_at_r3d

    or equivalently

    def rho_function(r3d, *function_args):
        return density_at_r3d

    :param function_args: a tuple (arg1, arg2, ...)
    :param use_pool: use multi-processing
    :return: deflection angles evaluated at Rvalues
    """

    args = []
    for k, ri in enumerate(Rvalues):
        args.append((ri, rho_function, function_args))

    if use_pool:
        from multiprocessing.pool import Pool
        pool = Pool(nproc)
        defangle = pool.map(deflection_point, args)
        pool.close()

    else:

        defangle = [deflection_point(args_i) for args_i in args]

    return np.array(defangle)

def deflection_multiprocessing(args):

    return deflection(*args)

def deflection_from_profile(Rvalues, rho_3D_array, r_evaluate):
    """
    :param three dimensional r coordinate
    :param rho_3D_array: the density at r
    :param r_evaluate: the coordinates at which to evaluate the deflection angles
    :return: the deflection angle at each Rcoordinate
    """
    rho_interp = interp1d(Rvalues, rho_3D_array)
    def _dummy_interp_function(x, *args, **kwargs):
        """
        Required since the density function must take some arguments, but interp1d takes only one argument
        """
        return rho_interp(x)

    return deflection(r_evaluate, _dummy_interp_function, function_args=None)

