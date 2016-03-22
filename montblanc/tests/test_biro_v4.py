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
import random
import time

import montblanc.factory
import montblanc.util as mbu

from montblanc.impl.biro.v4.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v4.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v4.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v4.gpu.RimeSumCoherencies import RimeSumCoherencies
from montblanc.impl.biro.v4.gpu.MatrixTranspose import MatrixTranspose

from montblanc.impl.biro.v4.cpu.SolverCPU import SolverCPU
from montblanc.pipeline import Pipeline
from montblanc.config import BiroSolverConfig as Options

def src_perms(slvr_cfg, permute_weights=False):
    """
    Permute the source types and return a SolverConfiguration suitable
    for use as input to the solver function/factory.

    Parameters:
        slvr_cfg : BiroSolverConfiguration
            Configuration containing other sensible defaults to include
            in the returned permutation.
            e.g. {'na': 14, 'ntime': 20, 'nchan': 48}

    { 'point': 0,  'gaussian': 00, 'sersic': 20 }
    { 'point': 0,  'gaussian': 20, 'sersic': 0  }
    { 'point': 20, 'gaussian': 0,  'sersic': 0  }
    { 'point': 0,  'gaussian': 20, 'sersic': 20 }
    { 'point': 20, 'gaussian': 20, 'sersic': 0  }
    { 'point': 20, 'gaussian': 0,  'sersic': 20 }
    { 'point': 20, 'gaussian': 20, 'sersic': 20 }

    >>> slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=20, nchan=48)
    >>> for p_slvr_cfg in src_perms(slvr_cfg, True)
    >>>     with solver(p_slvr_cfg) as slvr:
    >>>         slvr.solve()
    """

    if slvr_cfg is None:
        slvr_cfg = montblanc.rime_solver_cfg()

    from montblanc.src_types import SOURCE_VAR_TYPES

    src_types = SOURCE_VAR_TYPES.keys()

    count = 0

    weight_vector = [True, False] if permute_weights is True else [False]

    for wv in weight_vector:
        count = 0
        for p in itertools.product([0, 20], repeat=len(src_types)):
            # Nasty, but works to avoid the (0,0,0) case
            count += 1
            if count == 1:
                continue

            params = montblanc.rime_solver_cfg(**slvr_cfg)
            params[Options.WEIGHT_VECTOR] = wv
            src_dict = {s: p[i] for i,s in enumerate(src_types)}
            params[Options.SOURCES] = montblanc.sources(**src_dict)

            yield params

def solver(slvr_cfg, **kwargs):
    slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_TEST
    slvr_cfg[Options.VERSION] = Options.VERSION_FOUR
    slvr_cfg.update(kwargs)

    return montblanc.factory.rime_solver(slvr_cfg)

class TestBiroV4(unittest.TestCase):
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

    def EKBSqrt_test_impl(self, slvr, cmp=None):
        """ Type independent implementation of the EKBSqrt test """
        if cmp is None:
            cmp = {}

        # Make the beam cube sufficiently large to contain the
        # test values for the lm and pointing error coordinates
        # specified in BiroSolver.py
        S = 1

        slvr.set_beam_ll(-S)
        slvr.set_beam_lm(-S)
        slvr.set_beam_ul(S)
        slvr.set_beam_um(S)

        # Default parallactic angle is 0
        # Set it to 1 degree so that our
        # sources rotate through the cube.
        slvr.set_parallactic_angle(np.deg2rad(1))

        slvr_cpu = SolverCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        ekb_cpu = slvr_cpu.compute_ekb_sqrt_jones_per_ant()

        with slvr.context:
            ekb_gpu = slvr.jones_gpu.get()

        # Some proportion of values will be out due to
        # discrepancies on the CPU and GPU when computing
        # the E beam (See E_beam_test_impl below)
        proportion_acceptable = 1e-2
        d = np.invert(np.isclose(ekb_cpu, ekb_gpu, **cmp))
        incorrect = d.sum()
        proportion_incorrect = incorrect / float(d.size)
        self.assertTrue(proportion_incorrect < proportion_acceptable,
            ('Proportion of incorrect EKB %s '
            '(%d out of %d) '
            'is greater than the accepted tolerance %s.') %
                (proportion_incorrect,
                incorrect,
                d.size,
                proportion_acceptable))

        # Test that at a decent proportion of
        # the calculated EKB terms are non-zero
        non_zero = np.count_nonzero(ekb_cpu)
        non_zero_ratio = non_zero / float(ekb_cpu.size)
        self.assertTrue(non_zero_ratio > 0.85,
            'Non-zero EKB ratio is %f.' % non_zero)

    def test_EKBSqrt_float(self):
        """ Single precision EKBSqrt test  """

        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=20, nchan=64,
            sources=montblanc.sources(point=10, gaussian=10),
            dtype=Options.DTYPE_FLOAT,
            pipeline=Pipeline([RimeEBeam(), RimeBSqrt(), RimeEKBSqrt()]))

        with solver(slvr_cfg) as slvr:
            self.EKBSqrt_test_impl(slvr, cmp={'rtol': 1e-4})

    def test_EKBSqrt_double(self):
        """ Double precision EKBSqrt test """

        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=20, nchan=64,
            sources=montblanc.sources(point=10, gaussian=10),
            dtype=Options.DTYPE_DOUBLE,
            pipeline=Pipeline([RimeEBeam(), RimeBSqrt(), RimeEKBSqrt()]))

        with solver(slvr_cfg) as slvr:
            self.EKBSqrt_test_impl(slvr, cmp={'rtol': 1e-5})

    def sum_coherencies_test_impl(self, slvr, weight_vector=False, cmp=None):
        """ Type independent implementation of the coherency sum test """
        if cmp is None:
            cmp = {}

        slvr.set_sigma_sqrd(np.random.random(1)[0])

        # The pipeline for this test case doesn't
        # create the jones terms. Create some
        # random terms and transfer them to the GPU
        sh, dt = slvr.jones_shape, slvr.jones_dtype
        D = {'size': slvr.jones_shape}
        ekb_jones = np.random.random(size=sh).astype(dt) + \
            1j*np.random.random(size=sh).astype(dt)
        slvr.transfer_jones(ekb_jones)

        slvr_cpu = SolverCPU(slvr)

        # Call the GPU solver
        slvr.solve()

        # Check that the CPU and GPU visibilities
        # match each other
        ekb_per_bl = slvr_cpu.compute_ekb_jones_per_bl(ekb_jones)
        ekb_vis_cpu = slvr_cpu.compute_ekb_vis(ekb_per_bl)
        gekb_vis_cpu = slvr_cpu.compute_gekb_vis(ekb_vis_cpu)
        with slvr.context:
            gekb_vis_gpu = slvr.vis_gpu.get()

        self.assertTrue(np.allclose(gekb_vis_cpu, gekb_vis_gpu, **cmp))

        # Check that the chi squared sum terms
        # match each other
        chi_sqrd_sum_terms_cpu = slvr_cpu.compute_chi_sqrd_sum_terms(
            vis=gekb_vis_cpu, weight_vector=weight_vector)

        with slvr.context:
            chi_sqrd_sum_terms_gpu = slvr.chi_sqrd_result_gpu.get()

        self.assertTrue(np.allclose(chi_sqrd_sum_terms_cpu,
            chi_sqrd_sum_terms_gpu, **cmp))

        chi_sqrd_result_cpu = slvr_cpu.compute_biro_chi_sqrd(
            vis=gekb_vis_cpu, weight_vector=weight_vector)

        self.assertTrue(np.allclose(chi_sqrd_result_cpu, slvr.X2, **cmp))

    def test_sum_coherencies_float(self):
        """ Test the coherency sum float kernel """
        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=20, nchan=48,
            dtype=Options.DTYPE_FLOAT)

        for p_slvr_cfg in src_perms(slvr_cfg, permute_weights=True):
            wv = p_slvr_cfg[Options.WEIGHT_VECTOR]
            p_slvr_cfg['pipeline'] = Pipeline([RimeSumCoherencies(wv)])
            with solver(p_slvr_cfg) as slvr:

                self.sum_coherencies_test_impl(slvr,
                    cmp={'rtol': 1e-3},
                    weight_vector=wv)

    def test_sum_coherencies_double(self):
        """ Test the coherency sum double kernel """
        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=20, nchan=48,
            dtype=Options.DTYPE_DOUBLE)

        for p_slvr_cfg in src_perms(slvr_cfg, permute_weights=True):
            wv = p_slvr_cfg[Options.WEIGHT_VECTOR]
            p_slvr_cfg['pipeline'] = Pipeline([RimeSumCoherencies(wv)])
            with solver(p_slvr_cfg) as slvr:

                self.sum_coherencies_test_impl(slvr,
                    weight_vector=wv)

    def B_sqrt_test_impl(self, slvr, cmp=None):
        """ Type independent implementation of the B square root test """
        if cmp is None:
            cmp = {}

        # Call the GPU solver
        slvr.solve()

        # Calculate CPU version of the B sqrt matrix
        slvr_cpu = SolverCPU(slvr)
        b_sqrt_cpu = slvr_cpu.compute_b_sqrt_jones()

        # Calculate the GPU version of the B sqrt matrix
        with slvr.context:
            b_sqrt_gpu = slvr.B_sqrt_gpu.get()

        self.assertTrue(np.allclose(b_sqrt_cpu, b_sqrt_gpu, **cmp))

        # TODO: Replace with np.einsum
        # Pick 16 random points in the same and
        # check our square roots are OK for that
        nsrc, ntime, nchan = slvr.dim_global_size('nsrc', 'ntime', 'nchan')
        N = 16
        rand_srcs = [random.randrange(0, nsrc) for i in range(N)]
        rand_t = [random.randrange(0, ntime) for i in range(N)]
        rand_ch = [random.randrange(0, nchan) for i in range(N)]

        b_cpu = slvr_cpu.compute_b_jones()

        # Test that the square root of B
        # multiplied by itself yields B.
        # Also tests that the square root of B
        # is the Hermitian of the square root of B
        for src, t, ch in zip(rand_srcs, rand_t, rand_ch):
            B_sqrt = b_sqrt_cpu[src,t,ch].reshape(2,2)
            B = b_cpu[src,t,ch].reshape(2,2)
            self.assertTrue(np.allclose(B, np.dot(B_sqrt, B_sqrt)))
            self.assertTrue(np.all(B_sqrt == B_sqrt.conj().T))

    def test_B_sqrt_float(self):
        """ Test the B sqrt float kernel """

        slvr_cfg = montblanc.rime_solver_cfg(na=7, ntime=200, nchan=320,
            sources=montblanc.sources(point=10, gaussian=10),
            dtype=Options.DTYPE_FLOAT,
            pipeline=Pipeline([RimeBSqrt()]))

        with solver(slvr_cfg) as slvr:

            # This fails more often with an rtol of 1e-4
            self.B_sqrt_test_impl(slvr, cmp={'rtol': 1e-3})

    def test_B_sqrt_double(self):
        """ Test the B sqrt double kernel """

        slvr_cfg = montblanc.rime_solver_cfg(na=7, ntime=200, nchan=320,
            sources=montblanc.sources(point=10, gaussian=10),
            dtype=Options.DTYPE_DOUBLE,
            pipeline=Pipeline([RimeBSqrt()]))

        with solver(slvr_cfg) as slvr:

            self.B_sqrt_test_impl(slvr)

    def E_beam_test_impl(self, slvr, cmp=None):
        if cmp is None:
            cmp = {}

        # Make the beam cube sufficiently large to contain the
        # test values for the lm and pointing error coordinates
        # specified in BiroSolver.py
        S = 1

        slvr.set_beam_ll(-S)
        slvr.set_beam_lm(-S)
        slvr.set_beam_ul(S)
        slvr.set_beam_um(S)

        # Default parallactic angle is 0
        # Set it to 1 degree so that our
        # sources rotate through the cube.
        slvr.set_parallactic_angle(np.deg2rad(1))

        slvr_cpu = SolverCPU(slvr)
        E_term_cpu = slvr_cpu.compute_E_beam()

        slvr.solve()

        with slvr.context:
            E_term_gpu = slvr.jones_gpu.get()

        # After extensive debugging and attempts get a nice
        # solution, it has to be accepted that a certain
        # (small) proportion of values will be different.
        # These discrepancies are caused by taking the floor
        # of floating point grid positions that lie very close
        # to integral values. For example, the l position may
        # be 24.9999 on the CPU and 25.0000 on the GPU
        # (or vice-versa). I've made attempts to detect this
        # and get the CPU and the GPU to agree
        # e.g. np.abs(np.round(l) - l) < 1e-5
        # but whichever epsilon is chosen, one can still get
        # a discrepancy. e.g 24.990 vs 24.999

        # Hence, we choose a very low ratio of unnacceptable values
        proportion_acceptable = 1e-4
        d = np.invert(np.isclose(E_term_cpu, E_term_gpu, **cmp))
        incorrect = d.sum()
        proportion_incorrect = incorrect / float(d.size)
        self.assertTrue(proportion_incorrect < proportion_acceptable,
            ('Proportion of incorrect E beam values %s '
            '(%d out of %d) '
            'is greater than the accepted tolerance %s.') %
                (proportion_incorrect,
                incorrect,
                d.size,
                proportion_acceptable))

        # Test that at a decent proportion of
        # the calculated E terms are non-zero
        non_zero_E = np.count_nonzero(E_term_cpu)
        non_zero_E_ratio = non_zero_E / float(E_term_cpu.size)
        self.assertTrue(non_zero_E_ratio > 0.85,
            'Non-zero E-term ratio is {r}.'.format(r=non_zero_E_ratio))

    def E_beam_test_helper(self, beam_lw, beam_mh, beam_nud, dtype):
        slvr_cfg = montblanc.rime_solver_cfg(na=32, ntime=50, nchan=64,
            sources=montblanc.sources(point=10, gaussian=10),
            beam_lw=beam_lw, beam_mh=beam_mh, beam_nud=beam_nud,
            dtype=dtype,
            pipeline=Pipeline([RimeEBeam()]))

        with solver(slvr_cfg) as slvr:
            # Check that the beam cube dimensions are
            # correctly configured
            self.assertTrue(slvr.E_beam_shape == 
                (beam_lw, beam_mh, beam_nud, 4))

            self.E_beam_test_impl(slvr, cmp={'rtol': 1e-4})

    def test_E_beam_float(self):
        """ Test the E Beam float kernel """
        # Randomly configure the beam cube dimensions
        beam_lw = np.random.randint(50, 60)
        beam_mh = np.random.randint(50, 60)
        beam_nud = np.random.randint(50, 60)
        self.E_beam_test_helper(beam_lw, beam_mh, beam_nud,
            Options.DTYPE_FLOAT)

        beam_lw, beam_mh, beam_nud = 1, 1, 1
        self.E_beam_test_helper(beam_lw, beam_mh, beam_nud,
            Options.DTYPE_FLOAT)

    def test_E_beam_double(self):
        """ Test the E Beam double kernel """
        # Randomly configure the beam cube dimensions
        beam_lw = np.random.randint(50, 60)
        beam_mh = np.random.randint(50, 60)
        beam_nud = np.random.randint(50, 60)
        self.E_beam_test_helper(beam_lw, beam_mh, beam_nud,
            Options.DTYPE_DOUBLE)

        beam_lw, beam_mh, beam_nud = 1, 1, 1
        self.E_beam_test_helper(beam_lw, beam_mh, beam_nud,
            Options.DTYPE_DOUBLE)

    def test_sqrt_multiply(self):
        """
        Confirm that multiplying the square root
        of the brightness matrix into the
        per antenna jones terms results in the same
        jones matrices as multiplying the
        brightness matrix into the per baseline
        jones matrices.
         """

        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=10, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10),
            dtype=Options.DTYPE_DOUBLE,
            pipeline=Pipeline([]))

        with solver(slvr_cfg) as slvr:

            slvr_cpu = SolverCPU(slvr)
            nsrc, ntime, na, nbl, nchan = slvr.dim_global_size(
                'nsrc', 'ntime', 'na', 'nbl', 'nchan')

            # Calculate per baseline antenna pair indexes
            idx = slvr.get_ap_idx(src=True, chan=True)

            # Get the brightness matrix
            B = slvr_cpu.compute_b_jones()

            # Fill in the jones_cpu matrix with random values
            slvr.jones_cpu[:] = np.random.random(
                    size=slvr.jones_shape).astype(slvr.jones_dtype) + \
                np.random.random(
                    size=slvr.jones_shape).astype(slvr.jones_dtype)

            # Superfluous really, but makes below readable
            assert slvr.jones_shape == (nsrc, ntime, na, nchan, 4)

            # Get per baseline jones matrices from
            # the per antenna jones matrices
            J2, J1 = slvr.jones_cpu[idx]
            assert J1.shape == (nsrc, ntime, nbl, nchan, 4)
            assert J2.shape == (nsrc, ntime, nbl, nchan, 4)

            # Tile the brightness term over the baseline dimension
            # and transpose so that polarisations are last
            JB = np.tile(B[:,:,np.newaxis,:,:], (1,1,nbl,1,1))

            assert JB.shape == (nsrc, ntime, nbl, nchan, 4)

            # Calculate the first result using the classic equation
            # J2.B.J1^H
            res_one = SolverCPU.jones_multiply(J2, JB, J2.size/4)
            res_one = SolverCPU.jones_multiply_hermitian_transpose(
                res_one, J1, res_one.size/4)

            # Compute the square root of the
            # brightness matrix
            B_sqrt = slvr_cpu.compute_b_sqrt_jones(B)

            # Tile the brightness square root term over
            # the antenna dimension and transpose so that
            # polarisations are last
            JBsqrt = np.tile(B_sqrt[:,:,np.newaxis,:,:],
                (1,1,na,1,1))

            assert JBsqrt.shape == (nsrc, ntime, na, nchan, 4)

            # Multiply the square root of the brightness matrix
            # into the per antenna jones terms
            J = SolverCPU.jones_multiply(slvr.jones_cpu, JBsqrt, JBsqrt.size/4) \
                .reshape(nsrc, ntime, na, nchan, 4)

            # Get per baseline jones matrices from
            # the per antenna jones matrices
            J2, J1 = J[idx]
            assert J2.shape == (nsrc, ntime, nbl, nchan, 4)
            assert J1.shape == (nsrc, ntime, nbl, nchan, 4)

            # Calculate the first result using the optimised version
            # (J2.sqrt(B)).(J1.sqrt(B))^H == J2.sqrt(B).sqrt(B)^H.J1^H
            # == J2.sqrt(B).sqrt(B).J1^H
            # == J2.B.J1^H
            res_two = SolverCPU.jones_multiply_hermitian_transpose(
                J2, J1, J2.size/4)

            # Results from two different methods should be the same
            self.assertTrue(np.allclose(res_one, res_two))

    def test_jones_multiply(self):
        """ Verify jones matrix multiplication code against NumPy """

        # Generate random matrices
        def rmat(shape):
            return np.random.random(size=shape) + \
                np.random.random(size=shape)*1j

        N = 100
        shape = (100, 2,2)

        A, B = rmat(shape), rmat(shape)

        AM = [np.matrix(A[i,:,:]) for i in range(N)]
        BM = [np.matrix(B[i,:,:]) for i in range(N)]

        C = SolverCPU.jones_multiply(A, B, N)

        for Am, Bm, Cm in zip(AM, BM, C):
            assert np.allclose(Am*Bm, Cm)

        C = SolverCPU.jones_multiply_hermitian_transpose(A, B, N)

        for Am, Bm, Cm in zip(AM, BM, C):
            assert np.allclose(Am*Bm.H, Cm)

    def test_transpose(self):
        slvr_cfg = montblanc.rime_solver_cfg(na=14, ntime=10, nchan=16,
            sources=montblanc.sources(point=10, gaussian=10),
            weight_vector=True,
            pipeline=Pipeline([MatrixTranspose()]))


        with solver(slvr_cfg) as slvr:
            nsrc, nchan = slvr.dim_global_size('nsrc', 'nchan')

            slvr.register_array(
                name='matrix_in',
                shape=('nsrc', 'nchan'),
                dtype='ft',
                registrant='test_biro_v4')

            slvr.register_array(
                name='matrix_out',
                shape=('nchan', 'nsrc'),
                dtype='ft',
                registrant='test_biro_v4')

            matrix = np.random.random(
                size=(nsrc, nchan)).astype(slvr.ft)

            slvr.transfer_matrix_in(matrix)

            slvr.solve()

            with slvr.context:
                slvr.matrix_out_cpu = slvr.matrix_out_gpu.get()

            assert np.all(slvr.matrix_in_cpu == slvr.matrix_out_cpu.T)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBiroV4)
    unittest.TextTestRunner(verbosity=2).run(suite)
