import numpy as np
import pycuda.gpuarray as gpuarray

import montblanc

from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

from montblanc.impl.biro.v1.BiroSolver import BiroSolver

class TestSolver(BiroSolver):
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=0, dtype=np.float32, pipeline=None, **kwargs):

        # Store CPU arrays
        kwargs['store_cpu'] = True

        super(TestSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
			npsrc=npsrc, ngsrc=ngsrc, dtype=dtype, pipeline=pipeline, **kwargs)

        slvr = self
        na, nbl, nchan, ntime = slvr.na, slvr.nbl, slvr.nchan, slvr.ntime
        npsrc, ngsrc, nsrc = slvr.npsrc, slvr.ngsrc, slvr.nsrc
        ft, ct = slvr.ft, slvr.ct

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
            shape=slvr.uvw_shape, dtype=slvr.uvw_dtype)

        # Point source coordinates in the l,m,n (sky image) domain
        l=ft(np.random.random(nsrc)*0.1)
        m=ft(np.random.random(nsrc)*0.1)
        lm=shape_list([l,m], slvr.lm_shape, slvr.lm_dtype)

        # Brightness matrix for the point sources
        fI=ft(np.ones((ntime*nsrc,)))
        fQ=ft(np.random.random(ntime*nsrc)*0.5)
        fU=ft(np.random.random(ntime*nsrc)*0.5)
        fV=ft(np.random.random(ntime*nsrc)*0.5)
        alpha=ft(np.random.random(ntime*nsrc)*0.1)
        brightness = shape_list([fI,fQ,fU,fV,alpha],
            slvr.brightness_shape, slvr.brightness_dtype)

        # Gaussian shape matrix
        el = ft(np.random.random(ngsrc)*0.5)
        em = ft(np.random.random(ngsrc)*0.5)
        R = ft(np.ones(ngsrc)*100)
        gauss_shape = shape_list([el,em,R],
            slvr.gauss_shape_shape, slvr.gauss_shape_dtype)

        # Generate nchan frequencies/wavelengths
    	frequencies = ft(np.linspace(1e6,2e6,nchan))
        wavelength = ft(montblanc.constants.C/frequencies)
        slvr.set_ref_wave(wavelength[nchan//2])

        # Generate the antenna pointing errors
        point_errors = make_random(slvr.point_errors_shape,
            slvr.point_errors_dtype)

        # Generate the noise vector
        weight_vector = make_random(slvr.weight_vector_shape,
            slvr.weight_vector_dtype)

        # Copy the uvw, lm and brightness data to the gpu
        slvr.transfer_uvw(uvw)
        slvr.transfer_ant_pairs(slvr.get_default_ant_pairs())
        slvr.transfer_lm(lm)
        slvr.transfer_brightness(brightness)
        if ngsrc > 0: slvr.transfer_gauss_shape(gauss_shape)
        slvr.transfer_wavelength(wavelength)
        slvr.transfer_point_errors(point_errors)
        slvr.transfer_weight_vector(weight_vector)

        vis = make_random(slvr.vis_shape, slvr.vis_dtype) + \
            make_random(slvr.vis_shape, slvr.vis_dtype)*1j
        slvr.transfer_vis(vis)

        # The bayesian model
        assert slvr.bayes_data_shape == slvr.vis_shape
        bayes_data = make_random(slvr.bayes_data_shape,slvr.bayes_data_dtype) +\
            make_random(slvr.bayes_data_shape,slvr.bayes_data_dtype)*1j
        slvr.transfer_bayes_data(bayes_data)
        slvr.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])