import numpy as np
from BaseSharedData import *

class TestSharedData(GPUSharedData):
    def __init__(self, na=7, nsrc=10, nchan=8, ntime=5, dtype=np.float64, device=None):
        super(TestSharedData, self).__init__(na,nchan,ntime,nsrc,dtype,device)

        sd = self
        na,nbl,nchan,ntime,nsrc = sd.na,sd.nbl,sd.nchan,sd.ntime,sd.nsrc
        ft,ct = sd.ft,sd.ct

        # Baseline coordinates in the u,v,w (frequency) domain
        sd.uvw = np.array([ \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*3., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*2., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*1.], \
            dtype=ft)

        # Point source coordinates in the l,m,n (sky image) domain
        l=ft(np.random.random(nsrc)*0.1)
        m=ft(np.random.random(nsrc)*0.1)
        sd.lm=np.array([l,m], dtype=ft)

        # Brightness matrix for the point sources
        fI=ft(np.ones((nsrc,)))
        fQ=ft(np.random.random(nsrc)*0.5)
        fU=ft(np.random.random(nsrc)*0.5)
        fV=ft(np.random.random(nsrc)*0.5)
        alpha=ft(np.random.random(nsrc)*0.1)
        sd.brightness = np.array([fI,fQ,fU,fV,alpha], dtype=ft)

        # Generate nchan frequencies/wavelengths
    	frequencies = ft(np.linspace(1e6,2e6,nchan))
        sd.wavelength = 3e8/frequencies
        sd.set_ref_freq(frequencies[nchan//2])

        # Generate the antenna pointing errors
        sd.point_errors = np.random.random(np.product(sd.point_errors_shape))\
            .astype(ft).reshape(sd.point_errors_shape)

        # Copy the uvw, lm and brightness data to the gpu
        sd.transfer_uvw(sd.uvw)
        sd.transfer_ant_pairs(sd.get_default_ant_pairs())
        sd.transfer_lm(sd.lm)
        sd.transfer_brightness(sd.brightness)
        sd.transfer_wavelength(sd.wavelength)
        sd.transfer_point_errors(sd.point_errors)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        sd.keys = (np.arange(np.product(sd.jones_shape[:-1]))
            *sd.jones_shape[-1]).astype(np.int32)
        sd.keys_gpu = gpuarray.to_gpu(sd.keys)

        # Output visibility matrix
        assert np.product(sd.vis_shape) == np.product(sd.keys.shape)
        nviselements = np.product(sd.vis_shape)
        sd.vis = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
            .astype(ct).reshape(sd.vis_shape)
        sd.transfer_vis(sd.vis)

        # The bayesian model
        assert np.product(sd.bayes_model_shape) == np.product(sd.keys.shape)
        nbayes = np.product(sd.bayes_model_shape)
        sd.bayes_model = (np.random.random(nbayes) + np.random.random(nbayes)*1j)\
            .astype(ct).reshape(sd.bayes_model_shape)
        sd.transfer_bayes_model(sd.bayes_model)
        sd.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])

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
        assert phase.shape == (sd.nbl, sd.ntime, sd.nsrc)        	

        # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
        phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
        assert phase.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)        	

        # Dim nchan x ntime x nsrcs 
        power = np.power(sd.ref_freq/w[:,:,np.newaxis], sd.brightness[4])
        assert power.shape == (sd.nchan, sd.ntime, sd.nsrc)        	

        # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
        phase_term = power*np.exp(phase)
        assert phase_term.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)        	

        # Create the brightness matrix. Dim 4 x nsrcs
        brightness = sd.ct([
            sd.brightness[0]+sd.brightness[1] + 0j,     # fI+fQ + 0j
            sd.brightness[2] + 1j*sd.brightness[3],     # fU + fV*1j
            sd.brightness[2] - 1j*sd.brightness[3],     # fU - fV*1j
            sd.brightness[0]-sd.brightness[1] + 0j])        # fI-fQ + 0j
        assert brightness.shape == (4, sd.nsrc)

        # This works due to broadcast! Multiplies along
        # srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
        jones_cpu = (phase_term[np.newaxis,:,:,:,:]* \
            brightness[:,np.newaxis, np.newaxis, np.newaxis,:])\
            .reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))
        assert jones_cpu.shape == sd.jones_shape

        return jones_cpu

    def compute_bk_vis(self):
        return np.add.reduce(self.compute_bk_jones(), axis=4)        

    def compute_ebk_jones(self):
    	sd = self

        ap = sd.get_default_ant_pairs().reshape(2,sd.nbl*sd.ntime)
        ant0, ant1 = ap[0], ap[1]
        pe = sd.point_errors.reshape(2,sd.na*sd.ntime)

        ant0 = ant0*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)
        ant1 = ant1*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)

        d_p = pe[:,ant0].reshape(2,sd.nbl,sd.ntime)
        d_q = pe[:,ant1].reshape(2,sd.nbl,sd.ntime)

        # Broadcasting here produces, nbl x ntime x nsrc
        l_off = sd.lm[0] - d_p[0,:,:,np.newaxis]
        m_off = sd.lm[1] - d_p[1,:,:,np.newaxis]
        E_p = np.sqrt(l_off**2 + m_off**2)

        assert E_p.shape == (sd.nbl, sd.ntime, sd.nsrc)

        # Broadcasting here produces, nbl x nchan x ntime x nsrc
        E_p = E_p[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
        E_p = np.cos(sd.cos3_constant*E_p)**3

        assert E_p.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

        # Broadcasting here produces, nbl x ntime x nsrc
        l_off = sd.lm[0] - d_q[0,:,:,np.newaxis]
        m_off = sd.lm[1] - d_q[1,:,:,np.newaxis]
        E_q = np.sqrt(l_off**2 + m_off**2)

        assert E_q.shape == (sd.nbl, sd.ntime, sd.nsrc)

        # Broadcasting here produces, nbl x nchan x ntime x nsrc
        E_q = E_q[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
        E_q = np.cos(sd.cos3_constant*E_q)**3

        assert E_q.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

        return self.compute_bk_jones()*E_p*E_q

