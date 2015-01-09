import numpy
import numexpr
import pycuda
import os
import sys
import unittest

def print_versions():
    """
    Print the versions of software relied upon by montblanc.
    Inspired by numexpr testing suite.
    """
    print('-=' * 38)
    print('Python version:    %s' % sys.version)
    print('PyCUDA version:    %s' % pycuda.VERSION_TEXT)
    print("NumPy version:     %s" % numpy.__version__)
    print("Numexpr version:   %s" % numexpr.__version__)

    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print('Platform:          %s-%s' % (sys.platform, machine))

    print("AMD/Intel CPU?     %s" % numexpr.is_cpu_amd_intel)
    print("VML available?     %s" % numexpr.use_vml)

    if numexpr.use_vml:
        print("VML/MKL version:   %s" % numexpr.get_vml_version())
    print("Number of threads used by default: %d "
          "(out of %d detected cores)" % (numexpr.nthreads, numexpr.ncores))
    print('-=' * 38)

def suite():
#    from test_biro_v1 import TestBiroV1
    from test_biro_v2 import TestBiroV2
    from test_biro_v3 import TestBiroV3
    from test_base_solver import TestSolver
    from test_utils import TestUtils

    test_suite = unittest.TestSuite()
    niter = 1

    for n in range(niter):
#        test_suite.addTest(unittest.makeSuite(TestBiroV1))
        test_suite.addTest(unittest.makeSuite(TestBiroV2))
        test_suite.addTest(unittest.makeSuite(TestBiroV3))
        test_suite.addTest(unittest.makeSuite(TestSolver))
        test_suite.addTest(unittest.makeSuite(TestUtils))

    return test_suite

def test():
    print_versions()
    return unittest.TextTestRunner().run(suite())
