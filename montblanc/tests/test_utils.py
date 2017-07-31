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

import logging
import unittest
import numpy as np
import tempfile
import time

import montblanc
import montblanc.factory
import montblanc.util as mbu

class TestUtils(unittest.TestCase):
    """
    TestUtil class defining unit tests for
    montblanc's montblanc.util module
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)
        montblanc.setup_test_logging()

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_baseline_antenna_nrs(self):
        """ Test conversion between antenna and baseline numbers """

        def do_check(na, nbl, nbl_auto):
            # get nr baselines from nr of antenna
            self.assertTrue(mbu.nr_of_baselines(na) == nbl)
            self.assertTrue(mbu.nr_of_baselines(na, False) == nbl)
            self.assertTrue(mbu.nr_of_baselines(na, True) == nbl_auto)

            # get nr antenna from nr of baselines
            self.assertTrue(mbu.nr_of_antenna(nbl) == na)
            self.assertTrue(mbu.nr_of_antenna(nbl, False) == na)
            self.assertTrue(mbu.nr_of_antenna(nbl_auto, True) == na)

        do_check(7, 7*6//2, 7*8//2)                  # KAT7
        do_check(14, 14*13//2, 14*15//2)             # Westerbork
        do_check(27, 27*26//2, 27*28//2)             # VLA
        do_check(64, 64*63//2, 64*65//2)             # MeerKAT
        do_check(3500, 3500*3499//2, 3500*3501//2)   # SKA

    def test_random_like(self):
        """
        Test that the random_like function produces sensible data
        """

        # Try for floats and complex data
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            # Test random array creation with same
            # shape and type as existing array
            shape = (np.random.randint(1, 50), np.random.randint(1, 50))
            ary = np.empty(shape=shape, dtype=dtype)    
            random_ary = mbu.random_like(ary)

            # Test that that the shape and type is correct
            self.assertTrue(random_ary.shape == ary.shape)
            self.assertTrue(random_ary.dtype == dtype)

            # Test that we're getting complex data out
            if np.issubdtype(dtype, np.complexfloating):
                proportion_cplx = np.sum(np.iscomplex(random_ary)) / random_ary.size
                self.assertTrue(proportion_cplx > 0.9)

            # Test random array creation with supplied shape and type
            shape = (np.random.randint(1, 50), np.random.randint(1, 50))
            random_ary = mbu.random_like(shape=shape, dtype=dtype)

            # Test that that the shape and type is correct
            self.assertTrue(random_ary.shape == shape)
            self.assertTrue(random_ary.dtype == dtype)

            # Test that we're getting complex data out
            if np.issubdtype(dtype, np.complexfloating):
                proportion_cplx = np.sum(np.iscomplex(random_ary)) / random_ary.size
                self.assertTrue(proportion_cplx > 0.9)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)

