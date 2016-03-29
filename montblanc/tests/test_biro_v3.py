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
import time

import montblanc.factory

from montblanc.impl.biro.v2.cpu.SolverCPU import SolverCPU

from montblanc.config import BiroSolverConfig as Options

def solver(slvr_cfg, **kwargs):
    slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_TEST
    slvr_cfg[Options.VERSION] = Options.VERSION_THREE
    slvr_cfg.update(kwargs)

    return montblanc.factory.rime_solver(slvr_cfg)

class TestBiroV3(unittest.TestCase):
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

        slvr_cfg = montblanc.rime_solver_cfg(na=28, ntime=27, nchan=32,
            sources=montblanc.sources(point=50, gaussian=50),
            dtype=Options.DTYPE_DOUBLE)

        for wv in [True, False]:
            slvr_cfg[Options.WEIGHT_VECTOR] = wv
            with solver(slvr_cfg) as slvr:

                # Solve the RIME
                slvr.solve()

                # Compare CPU and GPU results
                slvr_cpu = SolverCPU(slvr)
                chi_sqrd_result_cpu = slvr_cpu.compute_biro_chi_sqrd(weight_vector=wv)
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp),
                    ('CPU (%s) and GPU (%s) '
                    'chi-squared value differ. '
                    'Failed for weight_vector=%s') %
                        (chi_sqrd_result_cpu, slvr.X2, wv))

    def test_budget(self):
        """
        Test that the CompositeSolver handles a memory budget
        """
        cmp = { 'rtol' : 1e-4}
        wv = True

        for t in [17, 27, 53]:
            slvr_cfg = montblanc.rime_solver_cfg(na=28, ntime=t, nchan=32,
                sources=montblanc.sources(point=50, gaussian=50),
                dtype=Options.DTYPE_FLOAT,
                weight_vector=wv, mem_budget=10*1024*1024, nsolvers=3)

            with solver(slvr_cfg) as slvr:

                # Solve the RIME
                slvr.solve()

                # Check that CPU and GPU results agree
                slvr_cpu = SolverCPU(slvr)
                chi_sqrd_result_cpu = slvr_cpu.compute_biro_chi_sqrd(weight_vector=wv)
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

                slvr.X2 = 0.0

                # Test that solving the RIME a second time produces
                # the same solution
                slvr.solve()
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp),
                    ('CPU (%s) and GPU (%s) '
                    'chi-squared values differ. ') %
                        (chi_sqrd_result_cpu, slvr.X2))

    #@unittest.skip('Skip timing test')
    def test_time(self):
        """ Test for timing purposes """

        wv = True

        slvr_cfg = montblanc.rime_solver_cfg(na=64, ntime=200, nchan=64,
            data_source=Options.DATA_SOURCE_DEFAULT, version=Options.VERSION_THREE,
            sources=montblanc.sources(point=50, gaussian=50),
            dtype=Options.DTYPE_FLOAT,
            weight_vector=wv)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:

            slvr.transfer_lm(slvr.lm_cpu)
            slvr.transfer_brightness(slvr.brightness_cpu)
            slvr.transfer_weight_vector(slvr.weight_vector_cpu)
            slvr.transfer_bayes_data(slvr.bayes_data_cpu)
            slvr.solve()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV3)
    unittest.TextTestRunner(verbosity=2).run(suite)
