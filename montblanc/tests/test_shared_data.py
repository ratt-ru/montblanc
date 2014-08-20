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

from montblanc.BaseSharedData import BaseSharedData

class TestSharedData(unittest.TestCase):
    """
    TestSharedData class defining unit tests for
    montblanc's BaseSharedData class
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
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

        name='uvw'
        shape=(3, sd.nbl, sd.ntime)
        gpu_name = name + '_gpu'
        cpu_name = name + '_cpu'
        shape_name = name + '_shape'
        dtype_name = name + '_dtype'

    	# Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

    	# Register the array
        sd.register_array(name=name, shape=shape,
    		dtype=np.float32, registrant='BaseSharedData')

    	# OK, The attributes should now exist
    	self.assertTrue(hasattr(sd, gpu_name))
    	self.assertTrue(hasattr(sd, cpu_name))

    	# uvw_gpu should be an gpuarray,
    	# while uvw_cpu should only have a descriptor backing it
    	# at this stage with no concrete NumPy array
    	self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
    	self.assertTrue(getattr(sd, cpu_name) is None)

    def test_register_array_create_cpu(self):
    	""" Test array registration requiring the creation of a CPU array """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

        name='uvw'
        shape=(3, sd.nbl, sd.ntime)
        gpu_name = name + '_gpu'
        cpu_name = name + '_cpu'
        shape_name = name + '_shape'
        dtype_name = name + '_dtype'

        # Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

    	# Register the array
        sd.register_array(name=name, shape=shape,
    		dtype=np.float32, registrant='BaseSharedData', cpu=True)

    	# OK, The attributes should now exist
    	self.assertTrue(hasattr(sd, gpu_name))
    	self.assertTrue(hasattr(sd, cpu_name))
    	self.assertTrue(not hasattr(sd, shape_name))
    	self.assertTrue(not hasattr(sd, dtype_name))

    	# uvw_gpu should return a gpuarray,
    	# uvw_cpu should return an ndarray.
    	self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
    	self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))

    def test_register_array_store_cpu(self):
        """ Test array registration deferrring to BaseSharedData store_cpu kwarg """
        sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10, store_cpu=True)

        name='uvw'
        shape=(3, sd.nbl, sd.ntime)
        gpu_name = name + '_gpu'
        cpu_name = name + '_cpu'
        shape_name = name + '_shape'
        dtype_name = name + '_dtype'

        # Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

        # Register the array
        sd.register_array(name=name, shape=shape,
            dtype=np.float32, registrant='BaseSharedData')

        # OK, The attributes should now exist
        self.assertTrue(hasattr(sd, gpu_name))
        self.assertTrue(hasattr(sd, cpu_name))
        self.assertTrue(not hasattr(sd, shape_name))
        self.assertTrue(not hasattr(sd, dtype_name))

        # uvw_gpu should return a gpuarray,
        # uvw_cpu should return an ndarray.
        self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))

    def test_register_array_create_cpu_not_gpu(self):
    	"""  Test array registration requiring a CPU array, but not a GPU array """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

        name='uvw'
        shape=(3, sd.nbl, sd.ntime)
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'
    	shape_name = name + '_shape'
    	dtype_name = name + '_dtype'

        # Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

    	# Register the array
    	sd.register_array(name=name, shape=shape,
    		dtype=np.float32, registrant='BaseSharedData', cpu=True, gpu=False)

    	# OK, The attributes should now exist
    	self.assertTrue(hasattr(sd, gpu_name))
    	self.assertTrue(hasattr(sd, cpu_name))
    	self.assertTrue(not hasattr(sd, shape_name))
    	self.assertTrue(not hasattr(sd, dtype_name))

    	# uvw_gpu should return a gpuarray,
    	# uvw_cpu should return an ndarray.
    	self.assertTrue(getattr(sd, gpu_name) is None)
    	self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))

    def test_register_array_create_existing(self):
        """  Test array registration of existing arrays """
        sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

        name='uvw'
        shape=(3, sd.nbl, sd.ntime)
        gpu_name = name + '_gpu'
        cpu_name = name + '_cpu'
        shape_name = name + '_shape'
        dtype_name = name + '_dtype'

        # Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

        # Register the array
        sd.register_array(name=name, shape=shape, dtype=np.float32,
            registrant='BaseSharedData', cpu=False, gpu=False)

        # OK, The attributes should now exist
        self.assertTrue(hasattr(sd, gpu_name))
        self.assertTrue(hasattr(sd, cpu_name))
        self.assertTrue(not hasattr(sd, shape_name))
        self.assertTrue(not hasattr(sd, dtype_name))

        # uvw_gpu should return None
        # uvw_cpu should return None.
        self.assertTrue(getattr(sd, gpu_name) is None)
        self.assertTrue(getattr(sd, cpu_name) is None)

        # Register the array again
        sd.register_array(name=name, shape=shape, dtype=np.float32,
            registrant='BaseSharedData', cpu=True, gpu=False)

        # uvw_gpu should return None
        # uvw_cpu should return an ndarray.
        self.assertTrue(getattr(sd, gpu_name) is None)
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))

        cpu_ary = np.random.random(size=shape).astype(np.float32)
        setattr(sd,cpu_name,cpu_ary)

        # Register the array again
        sd.register_array(name=name, shape=shape, dtype=np.float32,
            registrant='BaseSharedData', cpu=True, gpu=True)

        # uvw_gpu should return None
        # uvw_cpu should return an ndarray and should still be the same array
        self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))
        self.assertTrue(np.all(getattr(sd, cpu_name) == cpu_ary))

        gpu_ary = pycuda.curandom.rand(shape=shape,dtype=np.float32)
        setattr(sd,gpu_name,gpu_ary)

        # Register the array again
        sd.register_array(name=name, shape=shape, dtype=np.float32,
            registrant='BaseSharedData', cpu=True, gpu=True)

        # uvw_gpu should return a GPUArray and should still be the same array
        # uvw_cpu should return an ndarray and should still be the same array
        self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))
        self.assertTrue(np.all(getattr(sd, cpu_name) == cpu_ary))
        self.assertTrue(getattr(sd, gpu_name) == gpu_ary)

        # Check exception gets thrown if we try reregister with
        # a np.float64, a different type.
        with self.assertRaises(ValueError) as cm:
            # Register the array again
            sd.register_array(name=name, shape=shape, dtype=np.float64,
                registrant='BaseSharedData', cpu=True, gpu=True)

        self.assertTrue(cm.exception.message.find(
            'type float32 different to the supplied float64.') != -1)

        # uvw_gpu should return a GPUArray and should still be the same array
        # uvw_cpu should return an ndarray and should still be the same array
        self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))
        self.assertTrue(np.all(getattr(sd, cpu_name) == cpu_ary))
        self.assertTrue(getattr(sd, gpu_name) == gpu_ary)

        # Check exception gets thrown if we try reregister with
        # a diffferent shape.
        with self.assertRaises(ValueError) as cm:
            # Register the array again
            sd.register_array(name=name, shape=(2,sd.nbl, sd.ntime), dtype=np.float32,
                registrant='BaseSharedData', cpu=True, gpu=True)

        self.assertTrue(cm.exception.message.find(
            'shape (3, 3, 10) different to the supplied (2, 3, 10)') != -1)

        # uvw_gpu should return a GPUArray and should still be the same array
        # uvw_cpu should return an ndarray and should still be the same array
        self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
        self.assertTrue(isinstance(getattr(sd, cpu_name), np.ndarray))
        self.assertTrue(np.all(getattr(sd, cpu_name) == cpu_ary))
        self.assertTrue(getattr(sd, gpu_name) == gpu_ary)

    def test_register_array_shape_and_dtype(self):
    	""" Test array registration requiring shape and dtype attributes to be created """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

    	name='uvw'
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'
    	shape_name = name + '_shape'
    	dtype_name = name + '_dtype'

    	shape = (3,sd.nbl, sd.ntime)
    	dtype = np.complex64

        # Before registration, descriptors may not have been created
        # on the BaseSharedData class. If they have, check that
        # attributes associated with the sd instance are None
        if hasattr(BaseSharedData, cpu_name):
            self.assertTrue(getattr(sd, cpu_name) is None)
        if hasattr(BaseSharedData, gpu_name):
            self.assertTrue(getattr(sd, gpu_name) is None)

    	# Register the array
    	sd.register_array(name='uvw', shape=shape,
    		dtype=dtype, registrant='BaseSharedData',
    		shape_member=True, dtype_member=True)

    	# OK, The attributes should now exist
    	self.assertTrue(hasattr(sd, gpu_name))
    	self.assertTrue(hasattr(sd, cpu_name))
    	self.assertTrue(hasattr(sd, shape_name))
    	self.assertTrue(hasattr(sd, dtype_name))

    	# uvw_gpu should return a gpuarray,
    	# uvw_cpu should return an ndarray.
    	self.assertTrue(isinstance(getattr(sd, gpu_name), gpuarray.GPUArray))
    	self.assertTrue(getattr(sd, cpu_name) is None)
    	self.assertTrue(getattr(sd, shape_name) == shape)
    	self.assertTrue(getattr(sd, dtype_name) == dtype)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSharedData)
    unittest.TextTestRunner(verbosity=2).run(suite)