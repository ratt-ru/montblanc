import logging
import unittest
import numpy as np
import time
import sys

#import pycuda.autoinit
#import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

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
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'

    	# Before registration, the attributes do not exist
    	# Note that this should be the first test run, as the
    	# descriptors will be created on the BaseSharedData class.
    	self.assertTrue(not hasattr(sd, gpu_name))
    	self.assertTrue(not hasattr(sd, cpu_name))

    	# Register the array
    	sd.register_array(name='uvw', shape=(3, sd.nbl, sd.ntime),
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
    	""" """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

    	name='uvw'
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'
    	shape_name = name + '_shape'
    	dtype_name = name + '_dtype'

    	# Before registration, the attributes do not exist
    	#self.assertTrue(not hasattr(sd, gpu_name))
    	#self.assertTrue(not hasattr(sd, cpu_name))

    	# Register the array
    	sd.register_array(name='uvw', shape=(3, sd.nbl, sd.ntime),
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


    def test_register_array_create_cpu_not_gpu(self):
    	""" """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

    	name='uvw'
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'
    	shape_name = name + '_shape'
    	dtype_name = name + '_dtype'

    	# Before registration, the attributes do not exist
    	#self.assertTrue(not hasattr(sd, gpu_name))
    	#self.assertTrue(not hasattr(sd, cpu_name))

    	# Register the array
    	sd.register_array(name='uvw', shape=(3, sd.nbl, sd.ntime),
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

    def test_register_array_shape_and_dtype(self):
    	""" """
    	sd = BaseSharedData(na=3,nchan=32,ntime=10,npsrc=10,ngsrc=10)

    	name='uvw'
    	gpu_name = name + '_gpu'
    	cpu_name = name + '_cpu'
    	shape_name = name + '_shape'
    	dtype_name = name + '_dtype'

    	shape = (3,sd.nbl, sd.ntime)
    	dtype = np.complex64

    	# Before registration, the attributes do not exist
    	#self.assertTrue(not hasattr(sd, gpu_name))
    	#self.assertTrue(not hasattr(sd, cpu_name))

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