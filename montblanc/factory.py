#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import numpy as np

import pycuda.driver as cuda

import montblanc
import montblanc.util as mbu

from montblanc.config import (BiroSolverConfigurationOptions as Options)

from montblanc.pipeline import Pipeline

# PyCUDA device and context variables
__devices = None
__contexts = None

def get_contexts_per_device():
    """ Returns a list of CUDA contexts associated with each CUDA device """
    global __devices
    global __contexts

    # Create contexts for each device if they don't yet exist
    if __devices is None and __contexts is None:
        try:
            cuda.init()
        except Exception, e:
            raise RuntimeError('Montblanc was unable '
                'to initialise CUDA: %s' % repr(e))

        try:
            __devices = [cuda.Device(d) for d in range(cuda.Device.count())]
        except:
            raise RuntimeError('Montblanc was unable '
                'to create PyCUDA device objects: %s' % repr(e))

        try:
            __contexts = [d.make_context() for d in __devices]
        except:
            raise RuntimeError('Montblanc was unable '
                'to associate PyCUDA contexts '
                'with devices: %s' % repr(e))

        # Ask for an 8 byte shared memory config if we have Kepler
        for dev, ctx in zip(__devices,__contexts):
            if dev.compute_capability()[0] >= 3:
                ctx.set_shared_config(cuda.shared_config.EIGHT_BYTE_BANK_SIZE)

        # Pop each context off the stack
        for d in range(len(__devices)):
            cuda.Context.pop()

    return __contexts

def get_default_context():
    """ Get a default context """
    contexts = get_contexts_per_device()

    if not len(contexts) > 0:
        raise Exception, 'No CUDA devices availabe'

    return contexts[0]

def get_empty_pipeline(slvr_cfg):
    """ Get an empty pipeline object """
    return Pipeline([])

def get_base_solver(slvr_cfg):
    """ Get a basic solver object """

    # Get the default cuda context if none is provided
    if slvr_cfg.get('context', None) is None:
        slvr_cfg['context']=get_default_context()

    from montblanc.BaseSolver import BaseSolver

    return BaseSolver(slvr_cfg)

def create_rime_solver_from_ms(slvr_class_type, slvr_cfg):
    """ Initialise the supplied solver with measurement set data """
    version = slvr_cfg.get('version')

    # Complain if no MeasurementSet file was specified
    if Options.MS_FILE not in slvr_cfg:
        raise KeyError(('%s key is set to %s '
            'in the Solver Configuration, but '
            'no MeasurementSet file has been '
            'specified in the %s key') % (
                Options.DATA_SOURCE,
                Options.DATA_SOURCE_MS,
                Options.MS_FILE))

    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        from montblanc.impl.biro.v2.loaders import MeasurementSetLoader
    elif version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        from montblanc.impl.biro.v4.loaders import MeasurementSetLoader
    else:
        raise Exception, 'Incorrect version %s' % version

    with MeasurementSetLoader(slvr_cfg.get('msfile')) as loader:
        ntime, na, nchan = loader.get_dims()
        slvr_cfg[Options.NTIME] = ntime 
        slvr_cfg[Options.NA] = na 
        slvr_cfg[Options.NCHAN] = nchan 
        slvr = slvr_class_type(slvr_cfg)
        loader.load(slvr, slvr_cfg)
        return slvr

def create_rime_solver_from_test_data(slvr_class_type, slvr_cfg):
    """ Initialise the supplied solver with test data """
    version = slvr_cfg.get(Options.VERSION)

    slvr_cfg[Options.STORE_CPU] = True
    slvr = slvr_class_type(slvr_cfg)

    na, nbl, nchan, ntime = slvr.na, slvr.nbl, slvr.nchan, slvr.ntime
    npsrc, ngsrc, nssrc, nsrc = slvr.npsrc, slvr.ngsrc, slvr.nssrc, slvr.nsrc
    ft, ct = slvr.ft, slvr.ct

    # Curry the creation of a random array
    def make_random(shape,dtype):
        return np.random.random(size=shape).astype(dtype)

    def uvw_values(version):
        return np.arange(1,ntime*na+1)

        raise Exception, 'Invalid Version %s' % version

    # Baseline coordinates in the u,v,w (frequency) domain
    r = uvw_values(version)
    uvw = mbu.shape_list([30.*r, 25.*r, 20.*r],
            shape=slvr.uvw_shape, dtype=slvr.uvw_dtype)
    # Normalise Antenna 0
    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        uvw[:,:,0] = 0
    elif version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        uvw[:,0,:] = 0

    slvr.transfer_uvw(uvw)

    # Point source coordinates in the l,m,n (sky image) domain
    # 1e-4 ~ 20 arcseconds
    l=ft(np.random.random(nsrc)-0.5)*1e-4
    m=ft(np.random.random(nsrc)-0.5)*1e-4
    lm=mbu.shape_list([l,m], slvr.lm_shape, slvr.lm_dtype)
    slvr.transfer_lm(lm)

    # Brightness matrices
    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        B = np.empty(shape=slvr.brightness_shape, dtype=slvr.brightness_dtype)
        I, Q, U, V = B[0,:,:], B[1,:,:], B[2,:,:], B[3,:,:]
        alpha = B[4,:,:]
    elif version in [Options.VERSION_FOUR,Options.VERSION_FIVE]:
        # Stokes parameters
        # Need a positive semi-definite brightness
        # matrix for v4 and v5
        stokes = np.empty(shape=slvr.stokes_shape, dtype=slvr.stokes_dtype)
        I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
        alpha=np.empty_like(I)

    Q[:] = np.random.random(size=Q.shape)-0.5
    U[:] = np.random.random(size=U.shape)-0.5
    V[:] = np.random.random(size=V.shape)-0.5
    noise = np.random.random(size=(Q.shape))*0.1
    # Determinant of a brightness matrix
    # is I^2 - Q^2 - U^2 - V^2, noise ensures
    # positive semi-definite matrix
    I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)
    assert np.all(I**2 - Q**2 - U**2 - V**2 > 0.0)

    alpha[:] = np.random.random(size=I.shape)*0.1

    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        slvr.transfer_brightness(B)
    elif version in [Options.VERSION_FOUR,Options.VERSION_FIVE]:
        slvr.transfer_stokes(stokes)
        slvr.transfer_alpha(alpha)

    # E beam
    if version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        E_beam = make_random(slvr.E_beam_cpu.shape, slvr.E_beam_cpu.dtype) + \
            make_random(slvr.E_beam_cpu.shape, slvr.E_beam_cpu.dtype)*1j
        slvr.transfer_E_beam(E_beam)

    # G term
    if version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        G_term = make_random(slvr.G_term_cpu.shape, slvr.G_term_cpu.dtype) + \
            make_random(slvr.G_term_cpu.shape, slvr.G_term_cpu.dtype)*1j
        slvr.transfer_G_term(G_term)

    # Gaussian shape matrix
    el = ft(np.random.random(ngsrc)*0.5)
    em = ft(np.random.random(ngsrc)*0.5)
    R = ft(np.ones(ngsrc)*100)
    gauss_shape = mbu.shape_list([el,em,R],
            slvr.gauss_shape_shape, slvr.gauss_shape_dtype)
    if ngsrc > 0: slvr.transfer_gauss_shape(gauss_shape)

    # Sersic (exponential) shape matrix
    e1=ft(np.zeros((nssrc)))
    e2=ft(np.zeros((nssrc)))
    scale=ft(np.ones((nssrc)))
    sersic_shape = mbu.shape_list([e1,e2,scale],
            slvr.sersic_shape_shape, slvr.sersic_shape_dtype)
    if nssrc > 0: slvr.transfer_sersic_shape(sersic_shape)

    # Generate nchan frequencies/wavelengths
    frequencies = ft(np.linspace(1e6,2e6,nchan))
    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        wavelength = ft(montblanc.constants.C/frequencies)
        slvr.transfer_wavelength(wavelength)
        slvr.set_ref_wave(wavelength[nchan//2])
    elif version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        slvr.transfer_frequency(frequencies)
        slvr.set_ref_freq(frequencies[nchan//2])

    # Generate the antenna pointing errors
    point_errors = (make_random(slvr.point_errors_shape,
            slvr.point_errors_dtype)-0.5)*1e-5
    slvr.transfer_point_errors(point_errors)

    # Generate antenna scaling coefficients
    if version in [Options.VERSION_FOUR, Options.VERSION_FIVE]:
        antenna_scaling = make_random(slvr.antenna_scaling_shape,
            slvr.antenna_scaling_dtype)
        slvr.transfer_antenna_scaling(antenna_scaling)

    # Generate the noise vector
    weight_vector = make_random(slvr.weight_vector_shape,
            slvr.weight_vector_dtype)
    slvr.transfer_weight_vector(weight_vector)

    slvr.transfer_ant_pairs(slvr.get_default_ant_pairs())

    if version in [Options.VERSION_TWO, Options.VERSION_THREE]:
        # Generate random jones scalar values
        jones_scalar = make_random(slvr.jones_scalar_shape,
                slvr.jones_scalar_dtype)
        slvr.transfer_jones_scalar(jones_scalar)
    elif version in [Options.VERSION_FOUR,Options.VERSION_FIVE]:
        # Generate random jones scalar values
        jones = make_random(slvr.jones_shape,
                slvr.jones_dtype)
        slvr.transfer_jones(jones)

    vis = make_random(slvr.vis_shape, slvr.vis_dtype) + \
            make_random(slvr.vis_shape, slvr.vis_dtype)*1j
    slvr.transfer_vis(vis)

    # The bayesian model
    assert slvr.bayes_data_shape == slvr.vis_shape
    bayes_data = make_random(slvr.bayes_data_shape,slvr.bayes_data_dtype) +\
            make_random(slvr.bayes_data_shape,slvr.bayes_data_dtype)*1j
    slvr.transfer_bayes_data(bayes_data)
    slvr.set_sigma_sqrd((np.random.random(1)**2).astype(ft)[0])

    return slvr

def rime_solver(slvr_cfg):
    """ Factory function that produces a BIRO solver """

    data_source = slvr_cfg.get(Options.DATA_SOURCE, Options.DATA_SOURCE_MS)
    version = slvr_cfg.get(Options.VERSION, Options.DEFAULT_VERSION)

    # Get the default cuda context if none is provided
    if slvr_cfg.get('context', None) is None:
        slvr_cfg['context'] = get_default_context()

    # Figure out which version of BIRO solver we're dealing with.
    if version == Options.VERSION_TWO:
        from montblanc.impl.biro.v2.BiroSolver import BiroSolver
    elif version == Options.VERSION_THREE:
        from montblanc.impl.biro.v3.CompositeBiroSolver \
        import CompositeBiroSolver as BiroSolver
    elif version == Options.VERSION_FOUR:
        from montblanc.impl.biro.v4.BiroSolver import BiroSolver
    elif version == Options.VERSION_FIVE:
        from montblanc.impl.biro.v5.CompositeBiroSolver \
        import CompositeBiroSolver as BiroSolver
    else:
        raise Exception, 'Invalid version %s' % version

    if data_source == Options.DATA_SOURCE_MS:
        return create_rime_solver_from_ms(BiroSolver, slvr_cfg)
    elif data_source == Options.DATA_SOURCE_TEST:
        slvr_cfg[Options.STORE_CPU] = True
        return BiroSolver(slvr_cfg)
    elif data_source == Options.DATA_SOURCE_BIRO:
        return BiroSolver(slvr_cfg)
    else:
        raise Exception, 'Invalid type %s' % sd_type
