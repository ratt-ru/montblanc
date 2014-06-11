import numpy as np
import pycuda.gpuarray as gpuarray

import montblanc

from montblanc.BaseSharedData import GPUSharedData

class TestSharedData(GPUSharedData):
    def __init__(self, na=7, npsrc=10, nchan=8, ntime=5, ngsrc=0, dtype=np.float64, **kwargs):
        kwargs['store_cpu'] = True
        super(TestSharedData, self).__init__(na=na,nchan=nchan,ntime=ntime,
			npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)

        sd = self
        na,nbl,nchan,ntime = sd.na,sd.nbl,sd.nchan,sd.ntime
        npsrc, ngsrc, nsrc = sd.npsrc, sd.ngsrc, sd.nsrc
        ft,ct = sd.ft,sd.ct

        # Baseline coordinates in the u,v,w (frequency) domain
        uvw = np.array([ \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*3., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*2., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*1.], \
            dtype=ft)

        # Point source coordinates in the l,m,n (sky image) domain
        l=ft(np.random.random(nsrc)*0.1)
        m=ft(np.random.random(nsrc)*0.1)
        lm=np.array([l,m], dtype=ft)\
            .reshape(sd.lm_shape)

        # Brightness matrix for the point sources
        fI=ft(np.ones((nsrc,)))
        fQ=ft(np.random.random(nsrc)*0.5)
        fU=ft(np.random.random(nsrc)*0.5)
        fV=ft(np.random.random(nsrc)*0.5)
        alpha=ft(np.random.random(nsrc)*0.1)
        brightness = np.array([fI,fQ,fU,fV,alpha], dtype=ft)\
            .reshape(sd.brightness_shape)

        # Gaussian shape matrix
        el = ft(np.random.random(ngsrc)*0.5)
        em = ft(np.random.random(ngsrc)*0.5)
        R = ft(np.ones(ngsrc)*100)
        gauss_shape = np.array([el,em,R], dtype=ft)\
            .reshape(sd.gauss_shape_shape)

        # Generate nchan frequencies/wavelengths
    	frequencies = ft(np.linspace(1e6,2e6,nchan))
        wavelength = ft(montblanc.constants.C/frequencies)
        sd.set_ref_wave(wavelength[nchan//2])

        # Generate the antenna pointing errors
        point_errors = np.random.random(np.product(sd.point_errors_shape))\
            .astype(ft).reshape(sd.point_errors_shape)

        # Generate the noise vector
        noise_vector = np.random.random(np.product(sd.noise_vector_shape))\
            .astype(ft).reshape(sd.noise_vector_shape)

        # Copy the uvw, lm and brightness data to the gpu
        sd.transfer_uvw(uvw)
        sd.transfer_ant_pairs(sd.get_default_ant_pairs())
        sd.transfer_lm(lm)
        sd.transfer_brightness(brightness)
        if ngsrc > 0: sd.transfer_gauss_shape(gauss_shape)
        sd.transfer_wavelength(wavelength)
        sd.transfer_point_errors(point_errors)
        sd.transfer_noise_vector(noise_vector)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        sd.keys = (np.arange(np.product(sd.jones_shape[:-1]))
            *sd.jones_shape[-1]).astype(np.int32)
        sd.keys_gpu = gpuarray.to_gpu(sd.keys)

        # Output visibility matrix
        assert np.product(sd.vis_shape) == np.product(sd.keys.shape)
        nviselements = np.product(sd.vis_shape)
        vis = (np.random.random(nviselements) + np.random.random(nviselements)*1j)\
            .astype(ct).reshape(sd.vis_shape)
        sd.transfer_vis(vis)

        # The bayesian model
        assert np.product(sd.bayes_data_shape) == np.product(sd.keys.shape)
        nbayes = np.product(sd.bayes_data_shape)
        bayes_data = (np.random.random(nbayes) + np.random.random(nbayes)*1j)\
            .astype(ct).reshape(sd.bayes_data_shape)
        sd.transfer_bayes_data(bayes_data)
        sd.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])