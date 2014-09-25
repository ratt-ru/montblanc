import numpy as np

import montblanc
from montblanc.pipeline import Pipeline
from montblanc.node import Node, NullNode

VERSION_ONE = 'v1'
VERSION_TWO = 'v2'

MS_SD_TYPE = 'ms'
TEST_SD_TYPE = 'test'
BIRO_SD_TYPE = 'biro'

valid_biro_versions = [VERSION_ONE, VERSION_TWO]
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
		raise ValueError, 'Supplied version %s is not valid. ' \
			'Should be one of %s', (version, valid_biro_versions)

def check_biro_solver_type(sd_type):
	""" Throws an exception if the supplied shared data type is invalid """

	if sd_type not in valid_biro_solver_types:
		raise ValueError, 'Supplied shared data type %s is not valid. ' \
			'Should be one of %s', (sd_type, valid_biro_solver_types)

def get_empty_pipeline(**kwargs):
	return Pipeline([])

def get_bk_pipeline(**kwargs):
	from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
	from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce

	return Pipeline([RimeBK(), RimeJonesReduce()])

def get_bk_solver(sd_type=None, npsrc=1, ngsrc=0, dtype=np.float32,**kwargs):
	if kwargs.get('device', None) is None:
		import pycuda.autoinit
		kwargs['device']=pycuda.autoinit.device

	pipeline = get_bk_pipeline(**kwargs)

	check_msfile(kwargs.get('msfile', None))

	from montblanc.impl.biro.v1.MeasurementSetSolver import MeasurementSetSolver

	return MeasurementSetSolver(npsrc=npsrc, ngsrc=ngsrc,
		dtype=dtype, pipeline=pipeline, **kwargs)

def get_biro_pipeline(**kwargs):
	# Decide whether to use the weight vector
	use_weight_vector = kwargs.get('weight_vector', False)
	version = kwargs.get('version')

	if version == VERSION_ONE:
		from montblanc.impl.biro.v1.gpu.RimeEBK import RimeEBK
		from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce
		from montblanc.impl.biro.v1.gpu.RimeChiSquared import RimeChiSquared

		# Create a pipeline consisting of an EBK kernel, followed by a reduction,

		nodes = []
		# Add a node handling point sources, if any
		if kwargs.get('npsrc') > 0: nodes.append(RimeEBK(gaussian=False))
		# Add a node handling gaussian sources, if any
		if kwargs.get('ngsrc') > 0: nodes.append(RimeEBK(gaussian=True))

		# Followed by a reduction,
		# a chi squared difference between the Bayesian Model and the Visibilities
		# and a further reduction to produce the Chi Squared Value
		nodes.extend([RimeJonesReduce(),
			RimeChiSquared(weight_vector=use_weight_vector)])

		return Pipeline(nodes)
	elif version == VERSION_TWO:
		from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
		from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum

		# Create a pipeline consisting of an EK kernel, followed by a Gauss B Sum,
		nodes = [RimeEK(), RimeGaussBSum(weight_vector=use_weight_vector)]

		return Pipeline(nodes)

	raise Exception, 'Invalid Version %s' % version

def load_from_ms(**kwargs):
	check_msfile(kwargs.get('msfile',None))
	version = kwargs.get('version')

	if version == VERSION_TWO:
		from montblanc.impl.biro.v2.loaders import MeasurementSetLoader
		from montblanc.impl.biro.v2.BiroSolver import BiroSolver
	elif version == VERSION_ONE:
		from montblanc.impl.biro.v1.loaders import MeasurementSetLoader
		from montblanc.impl.biro.v1.BiroSolver import BiroSolver
	else:
		raise Exception, 'Incorrect version %s' % version

	with MeasurementSetLoader(kwargs.get('msfile')) as loader:
		ntime,na,nchan = loader.get_dims()
		slvr = BiroSolver(na=na,ntime=ntime,nchan=nchan,**kwargs)
		loader.load(slvr)
		return slvr

def create_biro_test_solver(**kwargs):
	version = kwargs.get('version')

	if version == VERSION_TWO:
		from montblanc.impl.biro.v2.BiroSolver import BiroSolver
	elif version == VERSION_ONE:
		from montblanc.impl.biro.v1.BiroSolver import BiroSolver
	else:
		raise Exception, 'Incorrect version %s' % version

	# Store CPU arrays
	kwargs['store_cpu'] = True

	slvr = BiroSolver(**kwargs)

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
		if version == VERSION_ONE:
			return np.arange(1,nbl*ntime+1)
		elif version == VERSION_TWO:
			return np.arange(1,ntime*na+1)
		
		raise Exception, 'Invalid Version %s' % version

	# Baseline coordinates in the u,v,w (frequency) domain
	r = uvw_values(version)
	uvw = shape_list([30.*r, 25.*r, 20.*r],
		shape=slvr.uvw_shape, dtype=slvr.uvw_dtype)
	# Normalise Antenna 0 for version two
	if version == VERSION_TWO: uvw[:,:,0] = 0
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

	if version == VERSION_ONE:
		# Generate random jones scalar values
		jones = make_random(slvr.jones_shape,
			slvr.jones_dtype)
		slvr.transfer_jones(jones)
	elif version == VERSION_TWO:
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

	if sd_type is None: sd_type=MS_SD_TYPE
	if version is None: version=VERSION_ONE

	check_biro_version(version)
	check_solver_type(sd_type)

	# Pack the supplied arguments into kwargs
	# so that we don't have to pass them around
	kwargs['npsrc'] = npsrc
	kwargs['ngsrc'] = ngsrc
	kwargs['dtype'] = dtype
	kwargs['version'] = version

	# Get the default cuda device if none is provided
	if kwargs.get('device', None) is None:
		import pycuda.autoinit
		kwargs['device']=pycuda.autoinit.device

	# Create a pipeline, if none is provided
	if kwargs.get('pipeline',None) is None:
		kwargs['pipeline'] = get_biro_pipeline(**kwargs)

	if sd_type == MS_SD_TYPE:
		return load_from_ms(**kwargs)
	elif sd_type == TEST_SD_TYPE:			
		return create_biro_test_solver(**kwargs)
	elif sd_type == BIRO_SD_TYPE:
		if version == VERSION_ONE:
			from montblanc.impl.biro.v1.BiroSolver import BiroSolver
		elif version == VERSION_TWO:
			from montblanc.impl.biro.v2.BiroSolver import BiroSolver
		else:
			raise 'Invalid version %s' % version

		return BiroSolver(**kwargs)

	raise Exception, 'Invalid Version %s' % version