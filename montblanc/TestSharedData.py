import numpy as np
import pycuda.gpuarray as gpuarray

import montblanc

from montblanc.BaseSharedData import GPUSharedData

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
        sd.wavelength = ft(montblanc.constants.C/frequencies)
        sd.set_ref_wave(sd.wavelength[nchan//2])

        # Generate the antenna pointing errors
        sd.point_errors = np.random.random(np.product(sd.point_errors_shape))\
            .astype(ft).reshape(sd.point_errors_shape)

        # Generate the noise vector
        sd.noise_vector = np.random.random(np.product(sd.noise_vector_shape))\
            .astype(ft).reshape(sd.noise_vector_shape)

        # Copy the uvw, lm and brightness data to the gpu
        sd.transfer_uvw(sd.uvw)
        sd.transfer_ant_pairs(sd.get_default_ant_pairs())
        sd.transfer_lm(sd.lm)
        sd.transfer_brightness(sd.brightness)
        sd.transfer_wavelength(sd.wavelength)
        sd.transfer_point_errors(sd.point_errors)
        sd.transfer_noise_vector(sd.noise_vector)

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
        assert np.product(sd.bayes_data_shape) == np.product(sd.keys.shape)
        nbayes = np.product(sd.bayes_data_shape)
        sd.bayes_data = (np.random.random(nbayes) + np.random.random(nbayes)*1j)\
            .astype(ct).reshape(sd.bayes_data_shape)
        sd.transfer_bayes_data(sd.bayes_data)
        sd.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])

    def compute_k_jones_scalar(self):
        """
        Computes the scalar K (phase) term of the RIME using numpy.

        Returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
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
        power = np.power(sd.ref_wave/w[:,:,np.newaxis], sd.brightness[4])
        assert power.shape == (sd.nchan, sd.ntime, sd.nsrc)        	

        # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
        phase_term = power*np.exp(phase)
        assert phase_term.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)        	


        return phase_term

    def compute_e_jones_scalar(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME

        returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self

        # Here we obtain our antenna pairs and pointing errors
        # TODO: The last dimensions are flattened to make indexing easier
        # later. There may be a more numpy way to do this but YOLO.
        ap = sd.get_default_ant_pairs().reshape(2,sd.nbl*sd.ntime)
        pe = sd.point_errors.reshape(2,sd.na*sd.ntime)

        # The flattened antenna pair array will look something like this.
        # It is based on 2 x nbl x ntime. Here we have 3 baselines and
        # 4 timesteps.
        #
        #            timestep
        #       0 1 2 3 0 1 2 3 0 1 2 3
        #
        # ant0: 0 0 0 0 0 0 0 0 1 1 1 1
        # ant1: 1 1 1 1 2 2 2 2 2 2 2 2

        # Create indexes into the pointing errors from the antenna pairs.
        # Pointing errors is 2 x na x ntime, thus each index will be
        # i = ANT*ntime + TIME. The TIME additions need to be padded by nbl.
        ant0 = ap[0]*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)
        ant1 = ap[1]*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)

        # Get the pointing errors for antenna p and q.
        d_p = pe[:,ant0].reshape(2,sd.nbl,sd.ntime)
        d_q = pe[:,ant1].reshape(2,sd.nbl,sd.ntime)

        # Compute the offsets for antenna 0 or p
        # Broadcasting here produces, nbl x ntime x nsrc
        l_off = sd.lm[0] - d_p[0,:,:,np.newaxis]
        m_off = sd.lm[1] - d_p[1,:,:,np.newaxis]
        E_p = np.sqrt(l_off**2 + m_off**2)

        assert E_p.shape == (sd.nbl, sd.ntime, sd.nsrc)

        # Broadcasting here produces, nbl x nchan x ntime x nsrc
        E_p = sd.beam_width*1e-9*E_p[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
        np.clip(E_p, np.finfo(sd.ft).min, sd.E_beam_clip, E_p)
        E_p = np.cos(E_p)**3

        assert E_p.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

        # Compute the offsets for antenna 1 or q
        # Broadcasting here produces, nbl x ntime x nsrc
        l_off = sd.lm[0] - d_q[0,:,:,np.newaxis]
        m_off = sd.lm[1] - d_q[1,:,:,np.newaxis]
        E_q = np.sqrt(l_off**2 + m_off**2)

        assert E_q.shape == (sd.nbl, sd.ntime, sd.nsrc)

        # Broadcasting here produces, nbl x nchan x ntime x nsrc
        E_q = sd.beam_width*1e-9*E_q[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
        np.clip(E_q, np.finfo(sd.ft).min, sd.E_beam_clip, E_q)
        E_q = np.cos(E_q)**3

        assert E_q.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

        return E_p*E_q

    def compute_ek_jones_scalar(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Return a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        return self.compute_k_jones_scalar()*\
            self.compute_e_jones_scalar()

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,nsrc) matrix of complex scalars.
        """
        sd = self
        # Create the brightness matrix. Dim 4 x nsrcs
        B = sd.ct([
            sd.brightness[0]+sd.brightness[1] + 0j,     # fI+fQ + 0j
            sd.brightness[2] + 1j*sd.brightness[3],     # fU + fV*1j
            sd.brightness[2] - 1j*sd.brightness[3],     # fU - fV*1j
            sd.brightness[0]-sd.brightness[1] + 0j])    # fI-fQ + 0j
        assert B.shape == (4, sd.nsrc)

        return B

    def compute_bk_jones(self):
        """
        Computes the BK term of the RIME.

        Returns a (4,nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self
        # Compute the K and B terms
        scalar_K = self.compute_k_jones_scalar()
        B = self.compute_b_jones()

        # This works due to broadcast! Multiplies phase and brightness along
        # srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
        jones_cpu = (scalar_K[np.newaxis,:,:,:,:]* \
            B[:,np.newaxis, np.newaxis, np.newaxis,:])#\
            #.reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))
        assert jones_cpu.shape == sd.jones_shape

        return jones_cpu 

    def compute_ebk_jones(self):
        """
        Computes the BK term of the RIME.

        Returns a (4,nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        return self.compute_bk_jones()*self.compute_e_jones_scalar()

    def compute_bk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar K term and the 2x2 B term.

        Returns a (4,nbl,nchan,ntime) matrix of complex scalars.
        """
        return np.add.reduce(self.compute_bk_jones(), axis=4)        
