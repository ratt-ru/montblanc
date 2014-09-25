import logging
import unittest
import numpy as np
import time
import sys

#import pycuda.autoinit
#import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom

import montblanc
import montblanc.ext.crimes

from montblanc.BaseSolver import BaseSolver

class TestSolver(unittest.TestCase):
    """
    TestSolver class defining unit tests for
    montblanc's BaseSolver class
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()*100))
        # Set up various things that aren't possible in PyCUDA
        montblanc.ext.crimes.setup_cuda()

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
    	with BaseSolver(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10) as slvr:
            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='BaseSolver')

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
        with BaseSolver(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='BaseSolver', cpu=True)

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
        """ Test array registration deferrring to BaseSolver store_cpu kwarg """
        with BaseSolver(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10, store_cpu=True) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='BaseSolver')

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
    	with BaseSolver(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape,
                dtype=np.float32, registrant='BaseSolver', cpu=True, gpu=False)

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
        with BaseSolver(na=14,nchan=32,ntime=10,npsrc=10,ngsrc=10) as slvr:

            name='uvw'
            shape=(3, slvr.nbl, slvr.ntime)
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='BaseSolver', cpu=False, gpu=False)

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
                registrant='BaseSolver', cpu=True, gpu=False)

            # uvw_gpu should return None
            # uvw_cpu should return an ndarray.
            self.assertTrue(getattr(slvr, gpu_name) is None)
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))

            cpu_ary = np.random.random(size=shape).astype(np.float32)
            setattr(slvr,cpu_name,cpu_ary)

            # Register the array again
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='BaseSolver', cpu=True, gpu=True)

            # uvw_gpu should return None
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))

            gpu_ary = pycuda.curandom.rand(shape=shape,dtype=np.float32)
            setattr(slvr,gpu_name,gpu_ary)

            # Register the array again
            slvr.register_array(name=name, shape=shape, dtype=np.float32,
                registrant='BaseSolver', cpu=True, gpu=True)

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
                    registrant='BaseSolver', cpu=True, gpu=True)

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
                    registrant='BaseSolver', cpu=True, gpu=True)

            self.assertTrue(cm.exception.message.find(
                'shape (3, 105, 10) different to the supplied (2, 105, 10)') != -1)

            # uvw_gpu should return a GPUArray and should still be the same array
            # uvw_cpu should return an ndarray and should still be the same array
            self.assertTrue(isinstance(getattr(slvr, gpu_name), gpuarray.GPUArray))
            self.assertTrue(isinstance(getattr(slvr, cpu_name), np.ndarray))
            self.assertTrue(np.all(getattr(slvr, cpu_name) == cpu_ary))
            self.assertTrue(getattr(slvr, gpu_name) == gpu_ary)

    def test_register_array_shape_and_dtype(self):
    	""" Test array registration requiring shape and dtype attributes to be created """
    	with BaseSolver(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10) as slvr:

            name='uvw'
            gpu_name = name + '_gpu'
            cpu_name = name + '_cpu'
            shape_name = name + '_shape'
            dtype_name = name + '_dtype'

            shape = (3,slvr.nbl, slvr.ntime)
            dtype = np.complex64

            # Before registration, descriptors may not have been created
            # on the BaseSolver class. If they have, check that
            # attributes associated with the slvr instance are None
            if hasattr(BaseSolver, cpu_name):
                self.assertTrue(getattr(slvr, cpu_name) is None)
            if hasattr(BaseSolver, gpu_name):
                self.assertTrue(getattr(slvr, gpu_name) is None)

            # Register the array
            slvr.register_array(name='uvw', shape=shape,
            dtype=dtype, registrant='BaseSolver',
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

    def test_auto_correlation(self):
        """
        Test the configuring our shared data object with auto auto-correlations
        provides the correct number of baselines
        """
        na, ntime, nchan, npsrc, ngsrc = 14, 5, 16, 2, 2

        # Should have 105 baselines for 14 antenna with auto-correlations on
        autocor = True
        with BaseSolver(na=na,ntime=5,nchan=16,npsrc=2,ngsrc=2,auto_correlations=autocor) as slvr:
            self.assertTrue(slvr.nbl == montblanc.nr_of_baselines(na,autocor))
            self.assertTrue(slvr.nbl == 105)

        # Should have 91 baselines for 14 antenna with auto-correlations on
        autocor = False
        with BaseSolver(na=na,ntime=5,nchan=16,npsrc=2,ngsrc=2,auto_correlations=autocor) as slvr:
            self.assertTrue(slvr.nbl == montblanc.nr_of_baselines(na,autocor))
            self.assertTrue(slvr.nbl == 91)

    def test_viable_timesteps(self):
        """
        Tests that various parts of the functionality for obtaining the
        number of viable timesteps work
        """
        ntime, nchan, npsrc, ngsrc = 5, 16, 2, 2
        with BaseSolver(na=14,ntime=ntime,nchan=nchan,npsrc=npsrc,ngsrc=ngsrc) as slvr:

            shape_one = (5,'ntime','nchan')
            shape_two = (10, 'nsrc')

            self.assertTrue(slvr.get_numeric_shape(shape_one) == (5,ntime,nchan))
            self.assertTrue(slvr.get_numeric_shape(shape_two) == (10,npsrc+ngsrc))

            self.assertTrue(slvr.get_numeric_shape(shape_one, ignore=['ntime']) == (5,nchan))
            self.assertTrue(slvr.get_numeric_shape(shape_two, ignore=['ntime']) == (10,npsrc+ngsrc))

            slvr.register_array(name='ary_one',shape=(5,'ntime', 'nchan'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            slvr.register_array(name='ary_two',shape=(10,'nsrc'), dtype=np.float64,
                registrant='test_solver', cpu=False, gpu=False)

            # How many timesteps can we accommodate with 2GB ?
            # Don't bother with the actual value, the assert in viable_timesteps
            # actually tests things quite well
            slvr.viable_timesteps(2*1024*1024*1024)

    def test_solver_factory(self):
        """ Test that the solver factory produces the correct types """
        import montblanc.factory

        with montblanc.factory.get_biro_solver(sd_type='biro',version='v1') as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v1.BiroSolver.BiroSolver)

        with montblanc.factory.get_biro_solver(sd_type='biro',version='v2') as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

        with montblanc.factory.get_biro_solver(sd_type='test',version='v1') as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v1.BiroSolver.BiroSolver)

        with montblanc.factory.get_biro_solver(sd_type='test',version='v2') as slvr:
            self.assertTrue(type(slvr) == montblanc.impl.biro.v2.BiroSolver.BiroSolver)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSolver)
    unittest.TextTestRunner(verbosity=2).run(suite)