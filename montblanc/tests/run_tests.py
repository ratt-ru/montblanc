#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import os
import sys
import unittest


def print_versions():
    import numpy
    import numexpr
    import montblanc
    import pycuda

    """
    Print the versions of software relied upon by montblanc.
    Inspired by numexpr testing suite.
    """
    print('-=' * 38)
    print('Python version:    %s' % sys.version)
    print('Montblanc version: %s' % montblanc.__version__)
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
    from test_rime_solver import TestSolver
    from test_utils import TestUtils
    from test_source_utils import TestSourceUtils

    from test_rime_v2 import TestRimeV2
    from test_rime_v4 import TestRimeV4
    from test_rime_v5 import TestRimeV5
    from test_rime_v5 import TestRimeV5
    from test_cmp_vis import TestCmpVis

    test_suite = unittest.TestSuite()
    niter = 1

    for n in range(niter):
        # The following three cases run really fast
        test_suite.addTest(unittest.makeSuite(TestSolver))
        test_suite.addTest(unittest.makeSuite(TestUtils))
        test_suite.addTest(unittest.makeSuite(TestSourceUtils))
        # Test recent code first, as it will be more likely to fail
        test_suite.addTest(unittest.makeSuite(TestCmpVis))
        test_suite.addTest(unittest.makeSuite(TestRimeV5))
        test_suite.addTest(unittest.makeSuite(TestRimeV4))
        test_suite.addTest(unittest.makeSuite(TestRimeV2))

    return test_suite

def test():
    print_versions()
    return unittest.TextTestRunner().run(suite())
