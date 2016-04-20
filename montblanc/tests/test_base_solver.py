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
import sys

import montblanc
import montblanc.factory
import montblanc.util as mbu

from montblanc.config import BiroSolverConfig as Options

def test_solver(slvr_cfg, **kwargs):
    slvr_cfg.update(kwargs)
    slvr = montblanc.factory.get_rime_solver(slvr_cfg)
    slvr.register_default_dimensions()

    return slvr

class TestSolver(unittest.TestCase):
    """
    TestSolver class defining unit tests for
    montblanc's BaseSolver class
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

    def test_viable_timesteps(self):
        """
        Tests that various parts of the functionality for obtaining the
        number of viable timesteps work
        """
        ntime, nchan, npsrc, ngsrc = 5, 16, 2, 2
        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=ntime, nchan=nchan,
            sources=montblanc.sources(point=2, gaussian=2))

        with test_solver(slvr_cfg) as slvr:

            slvr.register_array(name='ary_one',shape=(5,'ntime', 'nchan'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            slvr.register_array(name='ary_two',shape=(10,'nsrc'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            # How many timesteps can we accommodate with 2GB ?
            # Don't bother with the actual value, the assert in viable_timesteps
            # actually tests things quite well
            mbu.viable_timesteps(2*1024*1024*1024,
                slvr.arrays(), slvr.template_dict())

    def test_solver_factory(self):
        """ Test that the solver factory produces the correct types """
        slvr_cfg = montblanc.rime_solver_cfg(
            data_source=Options.DATA_SOURCE_DEFAULT, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

        slvr_cfg = montblanc.rime_solver_cfg(
            data_source=Options.DATA_SOURCE_TEST, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
