import numpy
import sys
from sidmpy.Profiles.deflection_angle_util import deflection
from sidmpy.Profiles.nfw_like import coreTNFWprofile

numpy.set_printoptions(threshold=sys.maxsize)
x = numpy.logspace(-3, 2, 50)
tau_values = numpy.arange(1, 31, 1)
beta_values = numpy.arange(0.0025, 1.005, 0.005)

def solve_deflection_angles(filename_out, use_pool=True, nproc=10):

    log_deflection_array = numpy.empty((len(beta_values), len(tau_values), len(x)))

    for i, beta in enumerate(beta_values):
        for j, tau in enumerate(tau_values):
            function_args = (1., 1., tau, beta)
            alpha = deflection(x, coreTNFWprofile, function_args, use_pool=use_pool, nproc=nproc)
            log_alpha = numpy.log10(alpha)
            log_deflection_array[i, j, :] = numpy.round(log_alpha, 3)

    with open(filename_out, 'w') as f:
        f.write('log_deflection_angle = np.')
        f.write(str(repr(log_deflection_array)) + '\n')
