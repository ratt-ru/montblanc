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

from montblanc.impl.rime.v4.cpu.CPUSolver import CPUSolver

from montblanc.config import RimeSolverConfig as Options

import montblanc.impl.rime.v4.RimeSolver as BSV4mod

def solver(slvr_cfg, **kwargs):
    slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_TEST
    slvr_cfg[Options.VERSION] = Options.VERSION_FIVE
    slvr_cfg.update(kwargs)

    return montblanc.factory.rime_solver(slvr_cfg)

class TestRimeV5(unittest.TestCase):
    """
    TestRimes class defining the unit test cases for montblanc
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)

        # Add a handler that outputs INFO level logging to file
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)

        montblanc.log.setLevel(logging.INFO)
        montblanc.log.handlers = [fh]

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_medium_budget(self):
        wv = True

        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=100, nchan=64,
            sources=montblanc.sources(point=100, gaussian=100, sersic=100),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=wv, nsolvers=3,
            source_batch_size=300,
            dtype=Options.DTYPE_DOUBLE)

        with solver(slvr_cfg) as slvr:
            montblanc.log.info(slvr)

            slvr.solve()


    def test_visibility_write_mode(self):
        """ Test visibility write mode """
        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=20, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10, sersic=10),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=True, dtype=Options.DTYPE_DOUBLE,
            vis_write=Options.VISIBILITY_WRITE_MODE_OVERWRITE)

        # Test that when the write mode is 'overwrite', multiple
        # calls to solve produce the same model visibilities
        with solver(slvr_cfg) as slvr:
            slvr.solve()
            vis = slvr.model_vis.copy()
            slvr.solve()
            assert np.allclose(vis, slvr.model_vis)

        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=20, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10, sersic=10),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=True, dtype=Options.DTYPE_DOUBLE,
            vis_write=Options.VISIBILITY_WRITE_MODE_SUM)

        # Test that when the write mode is 'sum', multiple
        # calls to solve produce a summation of model visibilities
        with solver(slvr_cfg) as slvr:
            slvr.solve()
            vis = slvr.model_vis.copy()
            slvr.solve()
            slvr.solve()
            slvr.solve()
            slvr.solve()
            assert np.allclose(5*vis, slvr.model_vis)

    def test_array_supply(self):
        """ Test that its possible to supply a custom array to the solver """
        uvw = np.zeros(shape=(20,27,3), dtype=np.float64)

        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=20, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10, sersic=10),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=True, dtype=Options.DTYPE_DOUBLE,
            array_cfg={'supplied' : {'uvw':uvw }})

        with solver(slvr_cfg) as slvr:
            # Test that all UVW elements are zero, as supplied by above array
            self.assertTrue(np.all(slvr.uvw == 0.0))
            # By contrast, lm is initialised with random test data
            # which is 90% non-zero
            self.assertTrue(np.count_nonzero(slvr.lm) > float(slvr.lm.size)*0.9)


        # Fall over on bad shape
        uvw = np.zeros(shape=(20,26,3), dtype=np.float64)

        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=20, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10, sersic=10),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=True, dtype=Options.DTYPE_DOUBLE,
            array_cfg={'supplied' : {'uvw':uvw }})

        with self.assertRaises(ValueError):
            with solver(slvr_cfg) as slvr:
                pass

        # Test that we can handle a single precision array
        # on a double precision solver
        visibilities = np.zeros(shape=(20,27*(27-1)//2,16,4), dtype=np.complex64)
        uvw = np.zeros(shape=(20,27,3), dtype=np.float32)

        slvr_cfg = montblanc.rime_solver_cfg(na=27, ntime=20, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10, sersic=10),
            beam_lw=50, beam_mh=50, beam_nud=50,
            weight_vector=True, dtype=Options.DTYPE_DOUBLE,
            array_cfg={'supplied' : {'model_vis':visibilities }})

        with solver(slvr_cfg) as slvr:
            self.assertTrue(slvr.model_vis.dtype == np.complex64)
            self.assertTrue(slvr.observed_vis.dtype == np.complex128)
            slvr.solve()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRimeV5)
    unittest.TextTestRunner(verbosity=2).run(suite)
