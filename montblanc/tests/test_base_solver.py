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

import pycuda.driver
import pycuda.gpuarray as gpuarray
import pycuda.curandom

import montblanc
import montblanc.factory
import montblanc.util as mbu

from montblanc.config import (SolverConfiguration,
    BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

class TestSolver(unittest.TestCase):
    """
    TestSolver class defining unit tests for
    montblanc's montblanc.factory.get_base_solver class
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

    def test_register_array_basic(self):
        """ """
        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            from montblanc.BaseSolver import BaseSolver

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='montblanc.factory.get_base_solver')

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))

            # uvw_gpu should be an gpuarray,
            # while uvw_cpu should only have a descriptor backing it
            # at this stage with no concrete NumPy array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(getattr(slvr, cpu_name) is None)

    def test_register_array_create_cpu(self):
        """ Test array registration requiring the creation of a CPU array """
        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:
            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            # Before registration, descriptors may not have been created
            # on the montblanc.factory.get_base_solver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(montblanc.factory.get_base_solver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(montblanc.factory.get_base_solver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='montblanc.factory.get_base_solver', cpu=True)

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))
            self.assertTrue(not hasattr(slvr, shape_name))
            self.assertTrue(not hasattr(slvr, dtype_name))

            # uvw_gpu should return a gpuarray,
            # uvw_cpu should return an ndarray.
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

    def test_register_array_store_cpu(self):
        """ Test array registration deferring to montblanc.factory.get_base_solver """

        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10),
            store_cpu=True)

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            # Before registration, descriptors may not have been created
            # on the montblanc.factory.get_base_solver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(montblanc.factory.get_base_solver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(montblanc.factory.get_base_solver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='montblanc.factory.get_base_solver')

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))
            self.assertTrue(not hasattr(slvr, shape_name))
            self.assertTrue(not hasattr(slvr, dtype_name))

            # uvw_gpu should return a gpuarray,
            # uvw_cpu should return an ndarray.
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

    def test_register_array_create_cpu_not_gpu(self):
        """  Test array registration requiring a CPU array, but not a GPU array """
        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            # Before registration, descriptors may not have been created
            # on the montblanc.factory.get_base_solver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(montblanc.factory.get_base_solver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(montblanc.factory.get_base_solver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='montblanc.factory.get_base_solver', cpu=True, gpu=False)

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))
            self.assertTrue(not hasattr(slvr, shape_name))
            self.assertTrue(not hasattr(slvr, dtype_name))

            # uvw_gpu should return a gpuarray,
            # uvw_cpu should return an ndarray.
            self.assertTrue(getattr(slvr, gpu_name) is None)
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

    def test_register_array_create_existing(self):
        """  Test array registration of existing arrays """
        slvr_cfg = SolverConfiguration(na=14, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr, \
            slvr.context as ctx:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            # Before registration, descriptors may not have been created
            # on the montblanc.factory.get_base_solver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(montblanc.factory.get_base_solver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(montblanc.factory.get_base_solver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='montblanc.factory.get_base_solver', cpu=False, gpu=False)

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))
            self.assertTrue(not hasattr(slvr, shape_name))
            self.assertTrue(not hasattr(slvr, dtype_name))

            # uvw_gpu should return None
            # uvw_cpu should return None.
            self.assertTrue(getattr(slvr, gpu_name) is None)
            self.assertTrue(getattr(slvr, cpu_name) is None)

            # Register the array again
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='montblanc.factory.get_base_solver', cpu=True, gpu=False)

            # uvw_gpu should return None
            # uvw_cpu should return an ndarray.
            self.assertTrue(getattr(slvr, gpu_name) is None)
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

            cpu_ary = np.random.random(size=shape).astype(np.float32)
            setattr(slvr,cpu_name,cpu_ary)

            # Register the array again
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='montblanc.factory.get_base_solver', cpu=True, gpu=True)

            # uvw_gpu should return None
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))

            gpu_ary = pycuda.curandom.rand(shape=shape,dtype=np.float32)
            setattr(slvr,gpu_name,gpu_ary)

            # Register the array again
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='montblanc.factory.get_base_solver', cpu=True, gpu=True)

            # uvw_gpu should return a GPUArray and should still be the same array
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))
            self.assertTrue(getattr(slvr, gpu_name) == gpu_ary)

            # Check exception gets thrown if we try reregister with
            # a np.float64, a different type.
            with self.assertRaises(ValueError) as cm:
                # Register the array again
                slvr.register_array(name=name, shape=shape, dtype=np.float64,
                    registrant='montblanc.factory.get_base_solver', cpu=True, gpu=True)

            self.assertTrue(cm.exception.message.find(
                'type float32 different to the supplied float64.') != -1)

            # uvw_gpu should return a GPUArray and should still be the same array
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))
            self.assertTrue(getattr(slvr, gpu_name) == gpu_ary)

            # Check exception gets thrown if we try reregister with
            # a different shape.
            with self.assertRaises(ValueError) as cm:
                # Register the array again
                slvr.register_array(name=name, shape=(2,slvr.nbl, slvr.ntime), dtype=np.float32,
                    registrant='montblanc.factory.get_base_solver', cpu=True, gpu=True)

            self.assertTrue(cm.exception.message.find(
                'shape (3, 91, 10) different to the supplied (2, 91, 10)') != -1)

            # uvw_gpu should return a GPUArray and should still be the same array
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))
            self.assertTrue(getattr(slvr, gpu_name) == gpu_ary)

    def test_register_array_shape_and_dtype(self):
        """ Test array registration requiring shape and dtype attributes to be created """
        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            name='uvw'
            gpu_name = mbu.gpu_name(name)
            cpu_name = mbu.cpu_name(name)
            shape_name = mbu.shape_name(name)
            dtype_name = mbu.dtype_name(name)

            shape = (3,slvr.nbl, slvr.ntime)
            dtype = np.complex64

            # Before registration, descriptors may not have been created
            # on the montblanc.factory.get_base_solver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(montblanc.factory.get_base_solver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(montblanc.factory.get_base_solver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name='uvw', shape=shape,
            dtype=dtype, registrant='montblanc.factory.get_base_solver',
            shape_member=True, dtype_member=True)

            # OK, The attributes should now exist
            self.assertTrue(hasattr(slvr, gpu_name))
            self.assertTrue(hasattr(slvr, cpu_name))
            self.assertTrue(hasattr(slvr, shape_name))
            self.assertTrue(hasattr(slvr, dtype_name))

            # uvw_gpu should return a gpuarray,
            # uvw_cpu should return an ndarray.
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(getattr(slvr, cpu_name) is None)
            self.assertTrue(getattr(slvr, shape_name) == shape)
            self.assertTrue(getattr(slvr, dtype_name) == dtype)

    def test_register_array_list(self):
        """ Test registration of arrays by list """
        D = [
            { 'name':'uvw', 'shape':(3,'ntime','na'),
                'dtype':np.float32, 'registrant':'test_base_solver',
                'cpu':True, 'gpu':True,
                'dtype_member':True, 'shape_member':True },
            { 'name':'brightness', 'shape':(5,'ntime','nsrc'),
                'dtype':np.complex64, 'registrant':'test_base_solver',
                'cpu':False, 'gpu':True,
                'dtype_member':True, 'shape_member':True },
            { 'name':'lm', 'shape':(2,'nsrc'),
                'dtype':np.float32, 'registrant':'test_base_solver',
                'cpu':True, 'gpu':False,
                'dtype_member':False, 'shape_member':False }
        ]

        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=montblanc.sources(point=10, gaussian=10))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            slvr.register_arrays(D)

            for a in D:
                name = a['name']
                gpu_name = mbu.gpu_name(name)
                cpu_name = mbu.cpu_name(name)
                shape_name = mbu.shape_name(name)
                dtype_name = mbu.dtype_name(name)

                if hasattr(montblanc.factory.get_base_solver, cpu_name):
                    if a['cpu'] is False:
                        self.assertTrue(getattr(slvr, cpu_name) is None)
                    else:
                        self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

                if hasattr(montblanc.factory.get_base_solver, gpu_name):
                    if a['gpu'] is False:
                        self.assertTrue(getattr(slvr, gpu_name) is None)
                    else:
                        self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))

                if a['dtype_member'] is True:
                    self.assertTrue(getattr(slvr,dtype_name) == a['dtype'])
                else:
                    self.assertTrue(not hasattr(slvr,dtype_name))

                if a['shape_member'] is True:
                    self.assertTrue(hasattr(slvr,shape_name))
                else:
                    self.assertTrue(not hasattr(slvr,shape_name))

    def test_register_property_list(self):
        """ Test registration of properties by list """
        D = [
            { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'test_base_solver', 'setter':True },
            { 'name':'beam_width','dtype':np.float32,
                'default':65, 'registrant':'test_base_solver', 'setter':True },
        ]

        sources = montblanc.sources(point=10, gaussian=10)
        slvr_cfg = SolverConfiguration(na=3, ntime=10, nchan=32,
            sources=sources)

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            slvr.register_properties(D)

            for a in D:
                name = a['name']
                self.assertTrue(hasattr(slvr,name))

                setter_name = mbu.setter_name(name)

                if a['setter']:
                    self.assertTrue(hasattr(slvr,setter_name))
                else:
                    self.assertFalse(hasattr(slvr,setter_name))

    def test_auto_correlation(self):
        """
        Test the configuring our solver object with auto auto-correlations
        provides the correct number of baselines
        """
        na, ntime, nchan, npsrc, ngsrc = 14, 5, 16, 2, 2

        # Should have 105 baselines for 14 antenna with auto-correlations on
        autocor = True
        slvr_cfg = SolverConfiguration(na=14, ntime=10, nchan=32,
            sources=montblanc.sources(point=2, gaussian=2),
            auto_correlations=autocor)

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:
            self.assertTrue(slvr.nbl == mbu.nr_of_baselines(na,autocor))
            self.assertTrue(slvr.nbl == 105)

        # Should have 91 baselines for 14 antenna with auto-correlations on
        autocor = False
        slvr_cfg = SolverConfiguration(na=14, ntime=10, nchan=32,
            sources=montblanc.sources(point=2, gaussian=2),
            auto_correlations=autocor)

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            self.assertTrue(slvr.nbl == mbu.nr_of_baselines(na,autocor))
            self.assertTrue(slvr.nbl == 91)

    def test_viable_timesteps(self):
        """
        Tests that various parts of the functionality for obtaining the
        number of viable timesteps work
        """
        ntime, nchan, npsrc, ngsrc = 5, 16, 2, 2
        slvr_cfg = SolverConfiguration(na=14, ntime=ntime, nchan=nchan,
            sources=montblanc.sources(point=2, gaussian=2))

        with montblanc.factory.get_base_solver(slvr_cfg) as slvr:

            slvr.register_array(name='ary_one',shape=(5,'ntime', 'nchan'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            slvr.register_array(name='ary_two',shape=(10,'nsrc'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            # How many timesteps can we accommodate with 2GB ?
            # Don't bother with the actual value, the assert in viable_timesteps
            # actually tests things quite well
            mbu.viable_timesteps(2*1024*1024*1024,
                slvr.arrays, slvr.get_properties())

    def test_solver_factory(self):
        """ Test that the solver factory produces the correct types """
        slvr_cfg = BiroSolverConfiguration(
            data_source=Options.DATA_SOURCE_DEFAULTS, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

        slvr_cfg = BiroSolverConfiguration(
            data_source=Options.DATA_SOURCE_TEST, version=Options.VERSION_TWO)

        with montblanc.factory.rime_solver(slvr_cfg) as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)
