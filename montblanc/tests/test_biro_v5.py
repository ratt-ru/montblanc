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
import montblanc.util as mbu

from montblanc.impl.biro.v4.cpu.SolverCPU import SolverCPU

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

import montblanc.impl.biro.v4.BiroSolver as BSV4mod

def solver(slvr_cfg, **kwargs):
    slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_TEST
    slvr_cfg[Options.VERSION] = Options.VERSION_FIVE
    slvr_cfg.update(kwargs)

    return montblanc.factory.rime_solver(slvr_cfg)

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

        slvr_cfg = BiroSolverConfiguration(na=14, ntime=27, nchan=32,
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
            with solver(na=28, npsrc=50, ngsrc=50, ntime=t, nchan=32,
                weight_vector=wv, mem_budget=10*1024*1024, nsolvers=3) as slvr:

                # Solve the RIME
                slvr.solve()

                # Check that CPU and GPU results agree
                chi_sqrd_result_cpu = SolverCPU(slvr).compute_biro_chi_sqrd(weight_vector=wv)
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

                slvr.X2 = 0.0

                # Test that solving the RIME a second time produces
                # the same solution
                slvr.solve()
                self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_big_budget(self):
        wv = True

        slvr_cfg = BiroSolverConfiguration(na=64, ntime=10000, nchan=32768,
            sources=montblanc.sources(point=10000, gaussian=10000, sersic=10000),
            beam_lw=1, beam_mh=1, beam_nud=1,
            weight_vector=wv, nsolvers=3,
            dtype=Options.DTYPE_DOUBLE)

        with solver(slvr_cfg) as slvr:
            slvr.solve()

    def test_medium_budget(self):
        wv = True

        slvr_cfg = BiroSolverConfiguration(na=27, ntime=100, nchan=64,
            sources=montblanc.sources(point=100, gaussian=100, sersic=100),
            beam_lw=1, beam_mh=1, beam_nud=1,
            weight_vector=wv, nsolvers=3,
            dtype=Options.DTYPE_DOUBLE)

        with solver(slvr_cfg) as slvr:
            print slvr

            slvr.solve()

    def test_smart_budget(self):
        wv = True

        slvr_cfg = BiroSolverConfiguration(na=28, ntime=27, nchan=128,
            sources=montblanc.sources(point=50, gaussian=50),
            beam_lw=1, beam_mh=1, beam_nud=1,
            weight_vector=wv, mem_budget=10*1024*1024, nsolvers=3,
            dtype=Options.DTYPE_DOUBLE)

        with solver(slvr_cfg) as slvr:

            A = copy.deepcopy(BSV4mod.A)
            T = slvr.template_dict()

            viable, MT = mbu.viable_dim_config(128*1024*1024,
                A, T, ['ntime', 'nbl&na', 'nchan'], 1)
            self.assertTrue(viable is True and len(MT) == 1 and
                MT['ntime'] == 1)

            viable, MT = mbu.viable_dim_config(8*1024*1024,
                A, T, ['ntime', 'nbl&na', 'nchan'], 1)
            self.assertTrue(viable is True and len(MT) == 3 and
                MT['ntime'] == 1 and
                MT['na'] == 1 and
                MT['nbl'] == 1)

            viable, MT = mbu.viable_dim_config(1*1024*1024,
                A, T, ['ntime', 'nbl&na', 'nchan'], 1)
            self.assertTrue(viable is True and len(MT) == 4 and
                MT['ntime'] == 1 and
                MT['na'] == 1 and
                MT['nbl'] == 1 and
                MT['nchan'] == 1)

            viable, MT = mbu.viable_dim_config(512*1024,
                A, T, ['ntime', 'nbl=6&na=3', 'nchan'], 1)
            self.assertTrue(viable is True and len(MT) == 4 and
                MT['ntime'] == 1 and
                MT['na'] == 3 and
                MT['nbl'] == 6 and
                MT['nchan'] == 1)

            viable, MT = mbu.viable_dim_config(
                1024, A, T, ['ntime', 'nbl&na'], 1)
            self.assertTrue(viable is False and len(MT) == 3 and
                MT['ntime'] == 1 and
                MT['na'] == 1 and
                MT['nbl'] == 1)

            # Try with 3 solvers.
            viable, MT = mbu.viable_dim_config(128*1024*1024,
                A, T, ['ntime', 'nbl&na', 'nchan'], 3)
            self.assertTrue(viable is True and len(MT) == 1 and
                MT['ntime'] == 1)


    @unittest.skip('Skip timing test')
    def test_time(self, cmp=None):
        """ Test for timing purposes """
        if cmp is None: cmp = {}

        for wv in [True]:
            with montblanc.factory.rime_solver('biro',version='v5',
                na=64,npsrc=50,ngsrc=50,ntime=200,nchan=64,weight_vector=wv) as slvr:

                slvr.transfer_lm(slvr.lm_cpu)
                slvr.transfer_brightness(slvr.brightness_cpu)
                slvr.transfer_weight_vector(slvr.weight_vector_cpu)
                slvr.transfer_bayes_data(slvr.bayes_data_cpu)
                slvr.solve()

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV5)
    unittest.TextTestRunner(verbosity=2).run(suite)
