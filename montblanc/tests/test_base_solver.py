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

from montblanc.config import RimeSolverConfig as Options

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

        # Add a handler that outputs INFO level logging to file
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)

        montblanc.log.setLevel(logging.INFO)
        montblanc.log.handlers = [fh]

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_solver_factory(self):
        """ Test that the solver factory produces the correct types """
        slvr_cfg = montblanc.rime_solver_cfg(
            data_source=Options.DATA_SOURCE_DEFAULT, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.rime.v2.RimeSolver.RimeSolver)

        slvr_cfg = montblanc.rime_solver_cfg(
            data_source=Options.DATA_SOURCE_TEST, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.rime.v2.RimeSolver.RimeSolver)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
