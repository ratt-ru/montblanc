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
from montblanc.pipeline import Pipeline

VERSION_ONE = 'v1'
VERSION_TWO = 'v2'
VERSION_THREE = 'v3'
DEFAULT_VERSION = VERSION_TWO

MS_SD_TYPE = 'ms'
TEST_SD_TYPE = 'test'
BIRO_SD_TYPE = 'biro'

deprecated_biro_versions = [VERSION_ONE]
valid_biro_versions = [VERSION_TWO, VERSION_THREE]
valid_biro_solver_types = [MS_SD_TYPE, TEST_SD_TYPE, BIRO_SD_TYPE]

def check_msfile(msfile):
	""" Check that the supplied msfile argument is a valid string """
	if msfile is None or not isinstance(msfile, str):
		raise TypeError, 'Invalid type %s specified for msfile' % type(msfile)

def check_solver_type(sd_type):
	""" Check that we have been supplied a valid solver type string """
	if sd_type not in valid_biro_solver_types:
		raise ValueError, ('Invalid BIRO solver type specified (%s).',
			'Should be one of %s') % (sd_type,valid_biro_solver_types)

def check_biro_version(version):
	""" Throws an exception if the supplied version is invalid """
	if version not in valid_biro_versions:
                if version in deprecated_biro_versions:
                    raise ValueError('Version %s is deprecated. ' %
                        version)
                else:
                    raise ValueError, 'Supplied version %s is not valid. ' \
                        'Should be one of %s', (version, valid_biro_versions)

def check_biro_solver_type(sd_type):
	""" Throws an exception if the supplied shared data type is invalid """

	if sd_type not in valid_biro_solver_types:
		raise ValueError, 'Supplied shared data type %s is not valid. ' \
			'Should be one of %s', (sd_type, valid_biro_solver_types)

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
        except:
            __devices, __contexts = None, None
            montblanc.log.critical('Unable to initialise CUDA', exc_info=True)

        try:
            __devices = [cuda.Device(d) for d in range(cuda.Device.count())]
        except:
            __devices, __contexts = None, None
            montblanc.log.critical('Unable to create devices', exc_info=True)

        try:
            __contexts = [d.make_context() for d in __devices]
        except:
            __devices, __contexts = None, None
            montblanc.log.critical('Unable to create contexts', exc_info=True)

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

def get_empty_pipeline(**kwargs):
	""" Get an empty pipeline object """
	return Pipeline([])

def get_bk_pipeline(**kwargs):
	""" Get a BK pipeline """
	from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
	from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce

	return Pipeline([RimeBK(), RimeJonesReduce()])

def get_base_solver(**kwargs):
	""" Get a basic solver object """
	pipeline = get_empty_pipeline(**kwargs)

	# Get the default cuda context if none is provided
	if kwargs.get('context', None) is None:
		kwargs['context']=get_default_context()

	from montblanc.BaseSolver import BaseSolver

	return BaseSolver(**kwargs)

def get_biro_pipeline(**kwargs):
	""" Get a BIRO pipeline """

	# Decide whether to use the weight vector
	use_weight_vector = kwargs.get('weight_vector', False)
	version = kwargs.get('version')

	if version == VERSION_TWO:
		from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
		from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum
		# Create a pipeline consisting of an EK kernel, followed by a Gauss B Sum,
		return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=use_weight_vector)])
	elif version == VERSION_THREE:
		from montblanc.impl.biro.v3.gpu.RimeEK import RimeEK
		from montblanc.impl.biro.v3.gpu.RimeGaussBSum import RimeGaussBSum
		# Create a pipeline consisting of an EK kernel, followed by a Gauss B Sum,
		return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=use_weight_vector)])

	raise Exception, 'Invalid Version %s' % version

def create_biro_solver_from_ms(slvr_class_type, **kwargs):
	""" Initialise the supplied solver with measurement set data """
	check_msfile(kwargs.get('msfile',None))
	version = kwargs.get('version')

	if version == VERSION_THREE:
		from montblanc.impl.biro.v3.loaders import MeasurementSetLoader
	elif version == VERSION_TWO:
		from montblanc.impl.biro.v2.loaders import MeasurementSetLoader
	else:
		raise Exception, 'Incorrect version %s' % version

	with MeasurementSetLoader(kwargs.get('msfile')) as loader:
		ntime,na,nchan = loader.get_dims()
		slvr = slvr_class_type(na=na,ntime=ntime,nchan=nchan,**kwargs)
		loader.load(slvr)
		return slvr

def create_biro_solver_from_test_data(slvr_class_type, **kwargs):
	""" Initialise the supplied solver with test data """
	version = kwargs.get('version')

	kwargs['store_cpu'] = True
	slvr = slvr_class_type(**kwargs)

	na, nbl, nchan, ntime = slvr.na, slvr.nbl, slvr.nchan, slvr.ntime
	npsrc, ngsrc, nsrc = slvr.npsrc, slvr.ngsrc, slvr.nsrc
	ft, ct = slvr.ft, slvr.ct

	# Curry the creation of a random array
	def make_random(shape,dtype):
	    return np.random.random(size=shape).astype(dtype)

	# Curry the shaping and casting of a list of arrays
	def shape_list(list,shape,dtype):
	    return np.array(list, dtype=dtype).reshape(shape)

	def uvw_values(version):
                return np.arange(1,ntime*na+1)

		raise Exception, 'Invalid Version %s' % version

	# Baseline coordinates in the u,v,w (frequency) domain
	r = uvw_values(version)
	uvw = shape_list([30.*r, 25.*r, 20.*r],
		shape=slvr.uvw_shape, dtype=slvr.uvw_dtype)
	# Normalise Antenna 0 for version two
	if version in [VERSION_TWO, VERSION_THREE]: uvw[:,:,0] = 0
	slvr.transfer_uvw(uvw)

	# Point source coordinates in the l,m,n (sky image) domain
	# 1e-4 ~ 20 arcseconds
	l=ft(np.random.random(nsrc)*1e-4)
	m=ft(np.random.random(nsrc)*1e-4)
	lm=shape_list([l,m], slvr.lm_shape, slvr.lm_dtype)
	slvr.transfer_lm(lm)

	# Brightness matrix for the point sources
	fI=ft(np.ones((ntime*nsrc,)))
	fQ=ft(np.random.random(ntime*nsrc)*0.5)
	fU=ft(np.random.random(ntime*nsrc)*0.5)
	fV=ft(np.random.random(ntime*nsrc)*0.5)
	alpha=ft(np.random.random(ntime*nsrc)*0.1)
	brightness = shape_list([fI,fQ,fU,fV,alpha],
	    slvr.brightness_shape, slvr.brightness_dtype)
	slvr.transfer_brightness(brightness)

	# Gaussian shape matrix
	el = ft(np.random.random(ngsrc)*0.5)
	em = ft(np.random.random(ngsrc)*0.5)
	R = ft(np.ones(ngsrc)*100)
	gauss_shape = shape_list([el,em,R],
		slvr.gauss_shape_shape, slvr.gauss_shape_dtype)
	if ngsrc > 0: slvr.transfer_gauss_shape(gauss_shape)

	# Generate nchan frequencies/wavelengths
	frequencies = ft(np.linspace(1e6,2e6,nchan))
	wavelength = ft(montblanc.constants.C/frequencies)
	slvr.transfer_wavelength(wavelength)
	slvr.set_ref_wave(wavelength[nchan//2])

	# Generate the antenna pointing errors
	point_errors = make_random(slvr.point_errors_shape,
		slvr.point_errors_dtype)
	slvr.transfer_point_errors(point_errors)

	# Generate the noise vector
	weight_vector = make_random(slvr.weight_vector_shape,
		slvr.weight_vector_dtype)
	slvr.transfer_weight_vector(weight_vector)

	slvr.transfer_ant_pairs(slvr.get_default_ant_pairs())

	if version in [VERSION_TWO, VERSION_THREE]:
		# Generate random jones scalar values
		jones_scalar = make_random(slvr.jones_scalar_shape,
			slvr.jones_scalar_dtype)
		slvr.transfer_jones_scalar(jones_scalar)

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

def get_biro_solver(sd_type=None, npsrc=1, ngsrc=0, dtype=np.float32,
	version=None, **kwargs):
	""" Factory function that produces a BIRO solver """

	if sd_type is None: sd_type=MS_SD_TYPE
	if version is None: version=DEFAULT_VERSION

	check_biro_version(version)
	check_solver_type(sd_type)

	# Pack the supplied arguments into kwargs
	# so that we don't have to pass them around
	kwargs['npsrc'] = npsrc
	kwargs['ngsrc'] = ngsrc
	kwargs['dtype'] = dtype
	kwargs['version'] = version

	# Get the default cuda context if none is provided
	if kwargs.get('context', None) is None:
		kwargs['context'] = get_default_context()

	# Create a pipeline, if none is provided
	if kwargs.get('pipeline',None) is None:
		kwargs['pipeline'] = get_biro_pipeline(**kwargs)

	# Figure out which version of BIRO solver we're dealing with.
	if version == VERSION_TWO:
		from montblanc.impl.biro.v2.BiroSolver import BiroSolver
	elif version == VERSION_THREE:
		from montblanc.impl.biro.v3.CompositeBiroSolver \
		import CompositeBiroSolver as BiroSolver
	else:
		raise Exception, 'Invalid version %s' % version

	if sd_type == MS_SD_TYPE:
		return create_biro_solver_from_ms(BiroSolver, **kwargs)
	elif sd_type == TEST_SD_TYPE:
		return create_biro_solver_from_test_data(BiroSolver, **kwargs)
	elif sd_type == BIRO_SD_TYPE:
		return BiroSolver(**kwargs)
	else:
		raise Exception, 'Invalid type %s' % sd_type