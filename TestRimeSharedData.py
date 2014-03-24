import numpy as np
from node import *

class TestRimeSharedData(SharedData):
    INIT = 'init'
    PRE = 'pre'
    EXEC = 'exec'
    POST = 'post'
    SHUTDOWN = 'shutdown'

    uvw_gpu = ArrayData()
    lm_gpu = ArrayData()
    brightness_gpu = ArrayData()

    na = Parameter(7)
    nbl = Parameter((7*6)/2)
    nchan = Parameter(32)
    nsrc = Parameter(200)
    ntime = Parameter(10)

    def __init__(self, na=7, nsrc=10, nchan=32, ntime=10,
            float_dtype=np.float64):
        super(TestRimeSharedData, self).__init__()
        self.set_params(na,nsrc,nchan,ntime,float_dtype)

    def set_params(self, na, nsrc, nchan, ntime, float_dtype):
        # Antenna, Baseline, Channel, Source and Timestep counts
        self.na = na
        self.nbl = (self.na**2 + self.na)/2
        self.nchan = nchan
        self.nsrc = nsrc
        self.ntime = ntime

        if float_dtype == np.float32:
            self.ct = np.complex64
        elif float_dtype == np.float64:
            self.ct = np.complex128
        else:
            raise TypeError, 'Must specify either np.float32 or np.float64 for float_dtype'

        self.ft = float_dtype

    def configure(self):
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        na, nbl = self.na, self.nbl
        nchan, ntime = self.nchan, self.ntime
        nsrc, ft, ct = self.nsrc, self.ft, self.ct

        self.stream = [cuda.Stream(), cuda.Stream()]

        self.event_names = [TestRimeSharedData.INIT, \
            TestRimeSharedData.PRE, \
            TestRimeSharedData.EXEC, \
            TestRimeSharedData.POST, \
            TestRimeSharedData.SHUTDOWN]
        self.nevents = len(self.event_names)

        # Baseline coordinates in the u,v,w (frequency) domain
        """
        self.uvw = np.array([ \
            np.ones(self.nbl, dtype=self.ft)*3., \
            np.ones(self.nbl, dtype=self.ft)*2., \
            np.ones(self.nbl, dtype=self.ft)*1.], \
            dtype=self.ft)
        """

        # UVW coordinates
        self.uvw = np.array([ \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*3., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*2., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*1.], \
            dtype=ft)

        # Point source coordinates in the l,m,n (sky image) domain
        l=ft(np.random.random(nsrc)*0.1)
        m=ft(np.random.random(nsrc)*0.1)
        self.lm=np.array([l,m], dtype=ft)

        # Brightness matrix for the point sources
        fI=ft(np.ones((nsrc,)))
        fQ=ft(np.random.random(nsrc)*0.5)
        fU=ft(np.random.random(nsrc)*0.5)
        fV=ft(np.random.random(nsrc)*0.5)
        alpha=ft(np.random.random(nsrc)*0.1)
        self.brightness = np.array([fI,fQ,fU,fV,alpha], dtype=ft)

        # Generate nchan frequencies/wavelengths
        self.wavelength = 3e8/ft(np.linspace(1e6,2e6,nchan))

        # Generate the antenna pointing errors
        self.point_errors = np.random.random(2*na*ntime)\
            .astype(ft).reshape((2, na, ntime))

        # Copy the uvw, lm and brightness data to the gpu
        self.uvw_gpu = gpuarray.to_gpu(self.uvw)
        self.lm_gpu = gpuarray.to_gpu(self.lm)
        self.brightness_gpu = gpuarray.to_gpu(self.brightness)
        self.wavelength_gpu = gpuarray.to_gpu(self.wavelength)
        self.point_errors_gpu = gpuarray.to_gpu(self.point_errors)

        # Output jones matrix
        self.jones_shape = (4,nbl,nchan,ntime,nsrc)
        self.jones_gpu = gpuarray.empty(self.jones_shape,dtype=ct)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        self.keys = (np.arange(np.product(self.jones_shape[:-1]))
            *self.jones_shape[-1]).astype(np.int32)
        self.keys_gpu = gpuarray.to_gpu(self.keys)

        # Output sum matrix
        self.sums_gpu = gpuarray.empty(self.keys.shape, dtype=ct)
