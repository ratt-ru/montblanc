import numpy as np
import pycuda.gpuarray as gpuarray

import montblanc

from montblanc.BaseSharedData import DEFAULT_NA
from montblanc.BaseSharedData import DEFAULT_NCHAN
from montblanc.BaseSharedData import DEFAULT_NTIME
from montblanc.BaseSharedData import DEFAULT_NPSRC
from montblanc.BaseSharedData import DEFAULT_NGSRC
from montblanc.BaseSharedData import DEFAULT_DTYPE

from montblanc.BiroSharedData import BiroSharedData

class TestSharedData(BiroSharedData):
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=0, dtype=np.float32, **kwargs):

        kwargs['store_cpu'] = True

        super(TestSharedData, self).__init__(na=na, nchan=nchan, ntime=ntime,
			npsrc=npsrc, ngsrc=ngsrc, dtype=dtype,**kwargs)

        sd = self
        na, nbl, nchan, ntime = sd.na, sd.nbl, sd.nchan, sd.ntime
        npsrc, ngsrc, nsrc = sd.npsrc, sd.ngsrc, sd.nsrc
        ft, ct = sd.ft, sd.ct

        # Curry the creation of a random array
        def make_random(shape,dtype):
            return np.random.random(size=shape).astype(dtype)

        # Curry the shaping and casting of a list of arrays
        def shape_list(list,shape,dtype):
            return np.array(list, dtype=dtype).reshape(shape)

        # Baseline coordinates in the u,v,w (frequency) domain
        uvw = shape_list([
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*3., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*2., \
            np.arange(1,nbl*ntime+1).reshape(nbl,ntime).astype(ft)*1.],
            shape=sd.uvw_shape, dtype=sd.uvw_dtype)

        # Point source coordinates in the l,m,n (sky image) domain
        l=ft(np.random.random(nsrc)*0.1)
        m=ft(np.random.random(nsrc)*0.1)
        lm=shape_list([l,m], sd.lm_shape, sd.lm_dtype)

        # Brightness matrix for the point sources
        fI=ft(np.ones((ntime*nsrc,)))
        fQ=ft(np.random.random(ntime*nsrc)*0.5)
        fU=ft(np.random.random(ntime*nsrc)*0.5)
        fV=ft(np.random.random(ntime*nsrc)*0.5)
        alpha=ft(np.random.random(ntime*nsrc)*0.1)
        brightness = shape_list([fI,fQ,fU,fV,alpha],
            sd.brightness_shape, sd.brightness_dtype)

        # Gaussian shape matrix
        el = ft(np.random.random(ngsrc)*0.5)
        em = ft(np.random.random(ngsrc)*0.5)
        R = ft(np.ones(ngsrc)*100)
        gauss_shape = shape_list([el,em,R],
            sd.gauss_shape_shape, sd.gauss_shape_dtype)

        # Generate nchan frequencies/wavelengths
    	frequencies = ft(np.linspace(1e6,2e6,nchan))
        wavelength = ft(montblanc.constants.C/frequencies)
        sd.set_ref_wave(wavelength[nchan//2])

        # Generate the antenna pointing errors
        point_errors = make_random(sd.point_errors_shape,
            sd.point_errors_dtype)

        # Generate the noise vector
        weight_vector = make_random(sd.weight_vector_shape,
            sd.weight_vector_dtype)

        # Copy the uvw, lm and brightness data to the gpu
        sd.transfer_uvw(uvw)
        sd.transfer_ant_pairs(sd.get_default_ant_pairs())
        sd.transfer_lm(lm)
        sd.transfer_brightness(brightness)
        if ngsrc > 0: sd.transfer_gauss_shape(gauss_shape)
        sd.transfer_wavelength(wavelength)
        sd.transfer_point_errors(point_errors)
        sd.transfer_weight_vector(weight_vector)

        # Create the key positions. This snippet creates an array
        # equal to the list of positions of the last array element timestep)
        sd.keys = (np.arange(np.product(sd.jones_shape[:-1]))
            *sd.jones_shape[-1]).astype(np.int32)
        sd.keys_gpu = gpuarray.to_gpu(sd.keys)

        # Output visibility matrix
        assert np.product(sd.vis_shape) == np.product(sd.keys.shape)

        vis = make_random(sd.vis_shape, sd.vis_dtype) + \
            make_random(sd.vis_shape, sd.vis_dtype)*1j
        sd.transfer_vis(vis)

        # The bayesian model
        assert sd.bayes_data_shape == sd.vis_shape
        bayes_data = make_random(sd.bayes_data_shape,sd.bayes_data_dtype) +\
            make_random(sd.bayes_data_shape,sd.bayes_data_dtype)*1j
        sd.transfer_bayes_data(bayes_data)
        sd.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])