import numpy as np
from BaseSharedData import *

class TestSharedData(GPUSharedData):
    def __init__(self, na=7, nsrc=10, nchan=8, ntime=5, dtype=np.float64, device=None):
        super(TestSharedData, self).__init__(na,nchan,ntime,nsrc,dtype,device)

        sd = self
        na,nbl,nchan,ntime,nsrc = sd.na,sd.nbl,sd.nchan,sd.ntime,sd.nsrc
        ft,ct = sd.ft,sd.ct

        # Baseline coordinates in the u,v,w (frequency) domain
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
    	frequencies = ft(np.linspace(1e6,2e6,nchan))
        self.wavelength = 3e8/frequencies
        self.set_reffreq(frequencies[nchan/2])

        # Generate the antenna pointing errors
        self.point_errors = np.random.random(2*na*ntime)\
            .astype(ft).reshape(self.point_errors_shape)

        # Copy the uvw, lm and brightness data to the gpu
        self.transfer_uvw(self.uvw)
        self.transfer_ant_pairs(self.get_default_ant_pairs())
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
        assert np.product(self.vis_shape) == np.product(self.keys.shape)
        nviselements = np.product(self.vis_shape)
        self.vis = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
            .astype(ct).reshape(self.vis_shape)
        self.transfer_vis(self.vis)

        # The bayesian model
        assert np.product(self.bayes_model_shape) == np.product(self.keys.shape)
        nbayes = np.product(self.bayes_model_shape)
        self.bayes_model = (np.random.random(nbayes) + np.random.random(nbayes)*1j)\
            .astype(ct).reshape(self.bayes_model_shape)
        self.transfer_bayes_model(self.bayes_model)
        self.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])

    def compute_bk_jones(self):
        sd = self
        # Repeat the wavelengths along the timesteps for now
        # dim nchan x ntime. 
        w = np.repeat(sd.wavelength,sd.ntime).reshape(sd.nchan, sd.ntime)

        # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
        n = np.sqrt(1. - sd.lm[0]**2 - sd.lm[1]**2) - 1.

        # u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
        phase = (np.outer(sd.uvw[0], sd.lm[0]) + \
            np.outer(sd.uvw[1], sd.lm[1]) + \
            np.outer(sd.uvw[2],n))\
                .reshape(sd.nbl, sd.ntime, sd.nsrc)

        # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
        phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
        # Dim nchan x ntime x nsrcs 
        power = np.power(sd.refwave/w[:,:,np.newaxis], sd.brightness[4])
        # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
        phase_term = power*np.exp(phase)

        # Create the brightness matrix. Dim 4 x nsrcs
        brightness = sd.ct([
            sd.brightness[0]+sd.brightness[1] + 0j,     # fI+fQ + 0j
            sd.brightness[2] + 1j*sd.brightness[3],     # fU + fV*1j
            sd.brightness[2] - 1j*sd.brightness[3],     # fU - fV*1j
            sd.brightness[0]-sd.brightness[1] + 0j])        # fI-fQ + 0j

        # This works due to broadcast! Multiplies along
        # srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
        jones_cpu = (phase_term[np.newaxis,:,:,:,:]* \
            brightness[:,np.newaxis, np.newaxis, np.newaxis,:])\
            .reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))

        return jones_cpu

    def compute_bk_vis(self):
        return np.add.reduce(self.compute_bk_jones(), axis=4)        
