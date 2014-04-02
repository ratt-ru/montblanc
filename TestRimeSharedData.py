import numpy as np
from BaseSharedData import *

class TestRimeSharedData(GPUSharedData):
    INIT = 'init'
    PRE = 'pre'
    EXEC = 'exec'
    POST = 'post'
    SHUTDOWN = 'shutdown'

    def __init__(self, na=7, nsrc=10, nchan=8, ntime=5, dtype=np.float64):
        super(TestRimeSharedData, self).__init__(na,nchan,ntime,nsrc,dtype)

    def configure(self):
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
        self.transfer_uvw(self.uvw)
        self.transfer_lm(self.lm)
        self.transfer_brightness(self.brightness)
        self.transfer_wavelength(self.wavelength)
        self.transfer_point_errors(self.point_errors)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        self.keys = (np.arange(np.product(self.jones_shape[:-1]))
            *self.jones_shape[-1]).astype(np.int32)
        self.keys_gpu = gpuarray.to_gpu(self.keys)

        # Output visibility matrix
        self.vis_shape = (4,nbl,nchan,ntime)
        assert np.product(self.vis_shape) == np.product(self.keys.shape)
        nvis = np.product(self.vis_shape)
        self.vis = (np.random.random(nvis) + np.random.random(nvis)*1j)\
            .astype(ct).reshape(self.vis_shape)
        self.transfer_vis(self.vis)

        # The bayesian model
        self.bayes_model_shape = self.vis_shape
        assert np.product(self.bayes_model_shape) == np.product(self.keys.shape)
        nbayes = np.product(self.bayes_model_shape)
        self.bayes_model = (np.random.random(nbayes) + np.random.random(nbayes)*1j)\
            .astype(ct).reshape(self.vis_shape)
        self.bayes_model_gpu = gpuarray.to_gpu(self.bayes_model)
        self.sigma_sqrd = (np.random.random(1)**2).astype(ft)[0]

        # Output chi squared terms
        self.chi_sqrd_shape = (nbl,nchan,ntime)
        self.chi_sqrd_gpu = gpuarray.empty(self.chi_sqrd_shape, dtype=ft)