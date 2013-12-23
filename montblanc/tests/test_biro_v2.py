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

import montblanc.factory

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum

from montblanc.impl.biro.v2.cpu.RimeCPU import RimeCPU
from montblanc.pipeline import Pipeline

def solver(**kwargs):
    return montblanc.factory.get_biro_solver('test',version='v2',**kwargs)

class TestBiroV2(unittest.TestCase):
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

    def EK_test_impl(self, slvr, cmp=None):
        """ Type independent implementaiton of the EK test """
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)

        rime_cpu = RimeCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ek_cpu = rime_cpu.compute_ek_jones_scalar_per_ant()
        with slvr.context: ek_gpu = slvr.jones_scalar_gpu.get()

        # Test that the jones CPU calculation matches that of the GPU calculation
        self.assertTrue(np.allclose(ek_cpu, ek_gpu, **cmp))

    def test_EK_float(self):
        """ Single precision EK test """
        with solver(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,
            dtype=np.float32,pipeline=Pipeline([RimeEK()])) as slvr:

            self.EK_test_impl(slvr)

    def test_EK_double(self):
        """ Double precision EK test """
        with solver(na=64,nchan=64,ntime=10,npsrc=20,ngsrc=20,
            dtype=np.float64,pipeline=Pipeline([RimeEK()])) as slvr:

            self.EK_test_impl(slvr)

    def gauss_B_sum_test_impl(self, slvr, weight_vector=False, cmp=None):
        if cmp is None: cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)
        slvr.set_sigma_sqrd(np.random.random(1)[0])

        rime_cpu = RimeCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ebk_vis_cpu = rime_cpu.compute_ebk_vis()
        with slvr.context: ebk_vis_gpu = slvr.vis_gpu.get()

        self.assertTrue(np.allclose(ebk_vis_cpu, ebk_vis_gpu, **cmp))

        chi_sqrd_result_cpu = rime_cpu.compute_biro_chi_sqrd(weight_vector=weight_vector)

        self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_gauss_B_sum_float(self):
        """ """
        for w in [True,False]:
            with solver(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20, dtype=np.float32,
                pipeline=Pipeline([RimeEK(), RimeGaussBSum(weight_vector=w)])) as slvr:

                self.gauss_B_sum_test_impl(slvr, weight_vector=w)

    def test_gauss_B_sum_double(self):
        """ """
        for w in [True,False]:
            with solver(na=14,nchan=48,ntime=20,npsrc=20,ngsrc=20, dtype=np.float64,
                pipeline=Pipeline([RimeEK(), RimeGaussBSum(weight_vector=w)])) as slvr:

                self.gauss_B_sum_test_impl(slvr, weight_vector=w)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)