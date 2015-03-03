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

import itertools
import logging
import unittest
import numpy as np
import time

import montblanc.factory

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK

from montblanc.impl.biro.v2.cpu.SolverCPU import SolverCPU
from montblanc.pipeline import Pipeline


def solver(**kwargs):
    """ Shorten the factory call for readability """
    return montblanc.factory.get_biro_solver('test', version='v2', **kwargs)


def src_perms(defaults, permute_weights=False):
    """
    Permute the source types and return a dictionary suitable
    for use as keyword arguments for the solver function/factory.

    Parameters:
        default : dictionary
            dictionary containing other sensible defaults to include
            in the returned permutation.
            e.g. {'na': 14, 'ntime': 20, 'nchan': 48}

    { 'npsrc': 0, 'ngsrc': 00, 'nssrc': 20}
    { 'npsrc': 0, 'ngsrc': 20, 'nssrc': 0}
    { 'npsrc': 20, 'ngsrc': 0, 'nssrc': 0}
    { 'npsrc': 0, 'ngsrc': 20, 'nssrc': 20}
    { 'npsrc': 20, 'ngsrc': 20, 'nssrc': 0}
    { 'npsrc': 20, 'ngsrc': 0, 'nssrc': 20}
    { 'npsrc': 20, 'ngsrc': 20, 'nssrc': 20}

    >>> for p in src_perms({'na': 14, 'ntime': 20, 'nchan': 48}, True)
    >>>     with solver(dtype=np.float32, **p) as slvr:
    >>>         slvr.solve()
    """

    if defaults is None:
        defaults = {}

    src_types = ['npsrc', 'ngsrc', 'nssrc']
    count = 0

    weight_vector = [True, False] if permute_weights is True else [False]

    for wv in weight_vector:
        count = 0
        for p in itertools.product([0, 20], repeat=len(src_types)):
            # Nasty, but works to avoid the (0,0,0) case
            count += 1
            if count == 1:
                continue

            params = defaults.copy()
            params['weight_vector'] = wv
            for i, s in enumerate(src_types):
                params[s] = p[i]

            yield params


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
        """ Type independent implementation of the EK test """
        if cmp is None:
            cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)

        slvr_cpu = SolverCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ek_cpu = slvr_cpu.compute_ek_jones_scalar_per_ant()
        with slvr.context:
            ek_gpu = slvr.jones_scalar_gpu.get()

        # Test that the jones CPU calculation matches
        # that of the GPU calculation
        self.assertTrue(np.allclose(ek_cpu, ek_gpu, **cmp))

    def test_EK_float(self):
        """ Single precision EK test  """
        for params in src_perms({'na': 64, 'nchan': 64, 'ntime': 10}, True):
            with solver(type=np.float32,
                        pipeline=Pipeline([RimeEK()]), **params) as slvr:

                self.EK_test_impl(slvr)

    def test_EK_double(self):
        """ Double precision EK test """
        for params in src_perms({'na': 64, 'nchan': 64, 'ntime': 10}, True):
            with solver(type=np.float64,
                        pipeline=Pipeline([RimeEK()]), **params) as slvr:

                self.EK_test_impl(slvr)

    def B_sum_test_impl(self, slvr, weight_vector=False, cmp=None):
        """ Type independent implementation of the B Sum test """
        if cmp is None:
            cmp = {}

        # This beam width produces reasonable values
        # for testing the E term
        slvr.set_beam_width(65*1e5)
        slvr.set_sigma_sqrd(np.random.random(1)[0])

        slvr_cpu = SolverCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ebk_vis_cpu = slvr_cpu.compute_ebk_vis()
        with slvr.context:
            ebk_vis_gpu = slvr.vis_gpu.get()

        self.assertTrue(np.allclose(ebk_vis_cpu, ebk_vis_gpu, **cmp))

        chi_sqrd_result_cpu = slvr_cpu.compute_biro_chi_sqrd(
            weight_vector=weight_vector)

        self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_B_sum_float(self):
        """ Test the B sum float kernel """
        for params in src_perms({'na': 14, 'ntime': 20, 'nchan': 48}, permute_weights=True):
            with solver(dtype=np.float32, **params) as slvr:
                self.B_sum_test_impl(slvr, params['weight_vector'], {'rtol': 1e-4})

    def test_B_sum_double(self):
        """ Test the B sum double kernel """
        for params in src_perms({'na': 14, 'ntime': 20, 'nchan': 48}, permute_weights=True):
            with solver(dtype=np.float64, **params) as slvr:
                self.B_sum_test_impl(slvr, params['weight_vector'])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV2)
    unittest.TextTestRunner(verbosity=2).run(suite)
