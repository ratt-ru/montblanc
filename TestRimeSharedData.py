import numpy as np
from node import *

class TestRimeSharedData(SharedData):
    INIT = 'init'
    PRE = 'pre'
    EXEC = 'exec'
    POST = 'post'
    SHUTDOWN = 'shutdown'

    uvw_gpu = ArrayData()
    lma_gpu = ArrayData()
    sky_gpu = ArrayData()

    na = Parameter(7)
    nbl = Parameter((7*6)/2)
    nchan = Parameter(32)
    nsrc = Parameter(200)
    ntime = Parameter(10)

    def __init__(self, na=7, nsrc=10, nchan=32, ntime=10,
            float_dtype=np.float64, complex_dtype=np.complex128):
        super(TestRimeSharedData, self).__init__()
        self.set_params(na,nsrc,nchan,ntime,float_dtype, complex_dtype)

    def set_params(self, na, nsrc, nchan, ntime, float_dtype, complex_dtype):
        # Antenna, Baseline, Channel, Source and Timestep counts
        self.na = na
        self.nbl = (self.na*(self.na-1))/2
        self.nchan = nchan
        self.nsrc = nsrc
        self.ntime = ntime

        self.ft = float_dtype
        self.ct = complex_dtype

    def configure(self):
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        self.stream = [cuda.Stream(), cuda.Stream()]

        self.event_names = [TestRimeSharedData.INIT, \
            TestRimeSharedData.PRE, TestRimeSharedData.EXEC, \
            TestRimeSharedData.POST, TestRimeSharedData.SHUTDOWN]
        self.nevents = len(self.event_names)

        # Baseline coordinates in the u,v,w (frequency) domain
        """
        self.uvw = np.array([ \
            np.ones(self.nbl, dtype=self.ft)*3., \
            np.ones(self.nbl, dtype=self.ft)*2., \
            np.ones(self.nbl, dtype=self.ft)*1.], \
            dtype=self.ft)
        """
        self.uvw = np.array([ \
            np.arange(1,self.nbl+1).astype(self.ft)*3., \
            np.arange(1,self.nbl+1).astype(self.ft)*2., \
            np.arange(1,self.nbl+1).astype(self.ft)*1.], \
            dtype=self.ft)

        # Point source coordinates in the l,m,n (sky image) domain
        l=self.ft(np.random.random(self.nsrc)*0.1)
        m=self.ft(np.random.random(self.nsrc)*0.1)
        alpha=self.ft(np.random.random(self.nsrc)*0.1)
        self.lma=np.array([l,m,alpha], \
            dtype=self.ft)

        # Brightness matrix for the point sources
        fI=self.ft(np.ones((self.nsrc,)))
        fV=self.ft(np.random.random(self.nsrc)*0.5)
        fU=self.ft(np.random.random(self.nsrc)*0.5)
        fQ=self.ft(np.random.random(self.nsrc)*0.5)
        self.sky = np.array([fI,fV,fU,fQ], \
            dtype=self.ft)

        # Generate nchan frequencies/wavelengths
        self.wavelength = 3e8/self.ft(np.linspace(1e6,2e6,self.nchan))

        # Generate the antenna pointing errors
        self.point_errors = np.random.random(2*self.na*self.ntime)\
            .astype(self.ft).reshape((2, self.na, self.ntime))

        # Copy the uvw, lma and sky data to the gpu
        self.uvw_gpu = gpuarray.to_gpu(self.uvw)
        self.lma_gpu = gpuarray.to_gpu(self.lma)
        self.sky_gpu = gpuarray.to_gpu(self.sky)
        self.wavelength_gpu = gpuarray.to_gpu(self.wavelength)
        self.point_errors_gpu = gpuarray.to_gpu(self.point_errors)

        # Output jones matrix
        self.jones_shape = (4,self.nbl,self.nchan,self.ntime,self.nsrc)
        self.jones_gpu = gpuarray.empty(self.jones_shape,dtype=self.ct)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        self.keys = (np.arange(np.product(self.jones_shape[:-1]))
            *self.jones_shape[-1]).astype(np.int32)
        self.keys_gpu = gpuarray.to_gpu(self.keys)

        # Output sum matrix
        self.sums_gpu = gpuarray.empty(self.keys.shape, dtype=self.ct)
