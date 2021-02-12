import numpy
import sys
from sidmpy.Profiles.deflection_angle_util import deflection
from sidmpy.Profiles.nfw_like import coreTNFWprofile

numpy.set_printoptions(threshold=sys.maxsize)
x = numpy.logspace(-3, 2, 50)
tau_values = numpy.arange(1, 31, 1)
beta_values = numpy.arange(0.0025, 1.005, 0.005)

def solve_deflection_angles(nproc=10):

    log_deflection_array = numpy.empty((len(beta_values), len(tau_values), len(x)))

    for i, beta in enumerate(beta_values):
        for j, tau in enumerate(tau_values):
            function_args = (x, 1., 1., tau, beta)
            deflection_args = (x, coreTNFWprofile, function_args, True, nproc)
            deflection_angle = deflection(*deflection_args)
            log_deflection_array[i, j, :] = numpy.log10(deflection_angle)

    return numpy.round(log_deflection_array, 3)
