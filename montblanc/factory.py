import numpy as np

VERSION_ONE = 'v1'
VERSION_TWO = 'v2'

MS_SD_TYPE = 'ms'
TEST_SD_TYPE = 'test'
BIRO_SD_TYPE = 'biro'

valid_biro_versions = [VERSION_ONE, VERSION_TWO]
valid_biro_solver_types = [MS_SD_TYPE, TEST_SD_TYPE, BIRO_SD_TYPE]

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

def get_bk_pipeline(**kwargs):
	from montblanc.pipeline import Pipeline
	from montblanc.impl.biro.v1.gpu.RimeBK import RimeBK
	from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce

	return Pipeline([RimeBK(), RimeJonesReduce()])

def get_biro_pipeline(npsrc=0, ngsrc=0, version=None, **kwargs):
	if version is None: version='v1'

	from montblanc.node import Node, NullNode
	from montblanc.pipeline import Pipeline

	check_biro_version(version)

	if not (npsrc + ngsrc > 0):
		raise ValueError, 'No point or gaussian sources have been specified!'

	# Decide whether to use the weight vector
	use_weight_vector = kwargs.get('weight_vector', False)

	if version == VERSION_ONE:
		from montblanc.impl.biro.v1.gpu.RimeEBK import RimeEBK
		from montblanc.impl.biro.v1.gpu.RimeJonesReduce import RimeJonesReduce
		from montblanc.impl.biro.v1.gpu.RimeChiSquared import RimeChiSquared

		# Create a pipeline consisting of an EBK kernel, followed by a reduction,

		nodes = []
		# Add a node handling point sources, if any
		if npsrc > 0: nodes.append(RimeEBK(gaussian=False))
		# Add a node handling gaussian sources, if any
		if ngsrc > 0: nodes.append(RimeEBK(gaussian=True))

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

def get_biro_solver(sd_type=None, npsrc=1, ngsrc=1, dtype=np.float32,
	version=None, **kwargs):
	if sd_type is None: sd_type=MS_SD_TYPE
	if version is None: version=VERSION_ONE

	check_biro_version(version)

	if kwargs.get('device', None) is None:
		import pycuda.autoinit
		kwargs['device']=pycuda.autoinit.device

	def check_msfile():
		msfile = kwargs.get('msfile', None)
		if msfile is None or not isinstance(msfile, str):
			raise TypeError, 'Invalid type %s specified for msfile' % type(msfile)

	if version == VERSION_ONE:
		if sd_type == MS_SD_TYPE:
			check_msfile()
			from montblanc.impl.biro.v1.MeasurementSetSolver import MeasurementSetSolver
			return MeasurementSetSolver(npsrc=npsrc, ngsrc=ngsrc,
				dtype=dtype,**kwargs)
		elif sd_type == TEST_SD_TYPE:			
			from montblanc.impl.biro.v1.TestSolver import TestSolver
			return TestSolver(npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)
		elif sd_type == BIRO_SD_TYPE:
			from montblanc.impl.biro.v1.BiroSolver import BiroSolver
			return BiroSolver(npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)

	if version == VERSION_TWO:
		if sd_type == MS_SD_TYPE:
			check_msfile()
			from montblanc.impl.biro.v2.MeasurementSetSolver import MeasurementSetSolver
			return MeasurementSetSolver(npsrc=npsrc, ngsrc=ngsrc,
				dtype=dtype,**kwargs)
		elif sd_type == TEST_SD_TYPE:			
			from montblanc.impl.biro.v2.TestSolver import TestSolver
			return TestSolver(npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)
		elif sd_type == BIRO_SD_TYPE:
			from montblanc.impl.biro.v2.BiroSolver import BiroSolver
			return BiroSolver(npsrc=npsrc,ngsrc=ngsrc,dtype=dtype,**kwargs)

	raise Exception, 'Invalid Version %s' % version
