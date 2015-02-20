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

import copy
import logging
import unittest
import numpy as np
import time

import montblanc.factory

from montblanc.impl.biro.v4.cpu.RimeCPU import RimeCPU

import montblanc.impl.biro.v4.BiroSolver as BSV4mod

def solver(**kwargs):
    return montblanc.factory.get_biro_solver('test',version='v5',**kwargs)

class TestBiroV5(unittest.TestCase):
    """
    TestRimes class defining the unit test cases for montblanc
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)

        # Add a handler that outputs INFO level logging
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)

        montblanc.log.addHandler(fh)
        montblanc.log.setLevel(logging.INFO)

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_basic(self):
        """ Basic Test """
        cmp = { 'rtol' : 1e-4}

        for wv in [True, False]:
            with solver(na=28, npsrc=50, ngsrc=50, ntime=27, nchan=32,
                weight_vector=wv) as slvr:

                # Solve the RIME
                slvr.solve()

                # Compare CPU and GPU results
                chi_sqrd_result_cpu = RimeCPU(slvr).compute_biro_chi_sqrd(weight_vector=wv)
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_budget(self):
        """
        Test that the CompositeSolver handles a memory budget, as well as
        dissimilar timesteps on the sub-solvers
        """
        cmp = { 'rtol' : 1e-4}
        wv = True

        with solver(na=28, npsrc=50, ngsrc=50, ntime=27, nchan=32,
            weight_vector=wv, mem_budget=10*1024*1024, nsolvers=3) as slvr:

            # Test for some variation in the sub-solvers
            self.assertTrue(slvr.solvers[0].ntime == 2)
            self.assertTrue(slvr.solvers[1].ntime == 2)
            self.assertTrue(slvr.solvers[2].ntime == 3)

            # Solve the RIME
            slvr.solve()

            # Check that CPU and GPU results agree
            chi_sqrd_result_cpu = RimeCPU(slvr).compute_biro_chi_sqrd(weight_vector=wv)
            self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_smart_budget(self):
        wv = True

        with solver(na=28, npsrc=50, ngsrc=50, ntime=27, nchan=32,
            weight_vector=wv, mem_budget=10*1024*1024, nsolvers=3) as slvr:

            A = copy.deepcopy(BSV4mod.A)

            viable, P = slvr.viable_dim_config(
                10*1024*1024, A,  ['ntime', 'nbl', 'nchan'], 1, True)
            print viable, P

            viable, P = slvr.viable_dim_config(
                1*1024*1024, A, ['ntime', 'nbl', 'nchan'], 1, True)
            print viable, P

            viable, P = slvr.viable_dim_config(
                512*1024, A, ['ntime', 'nbl', 'nchan'], 1, True)
            print viable, P

            viable, P = slvr.viable_dim_config(
                1024, A, ['ntime', 'nbl'], 1, True)
            print viable, P


    @unittest.skip('Problem size causes allocation failures during run of '
        'entire test suite.')
    def test_time(self, cmp=None):
        """ Test for timing purposes """
        if cmp is None: cmp = {}

        for wv in [True]:
            with montblanc.factory.get_biro_solver('biro',version='v3',
                na=64,npsrc=50,ngsrc=50,ntime=200,nchan=64,weight_vector=wv) as slvr:

                slvr.transfer_lm(slvr.lm_cpu)
                slvr.transfer_brightness(slvr.brightness_cpu)
                slvr.transfer_weight_vector(slvr.weight_vector_cpu)
                slvr.solve()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV5)
    unittest.TextTestRunner(verbosity=2).run(suite)
