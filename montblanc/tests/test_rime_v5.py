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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRimeV5)
    unittest.TextTestRunner(verbosity=2).run(suite)
