import numpy as np

from node import Node, NullNode
from pipeline import Pipeline
from BaseSharedData import GPUSharedData

from RimeBKFloat import RimeBKFloat
from RimeEBKFloat import RimeEBKFloat
from RimeJonesReduce import RimeJonesReduceFloat
from RimeChiSquaredFloat import RimeChiSquaredFloat
from RimeChiSquaredReduceFloat import RimeChiSquaredReduceFloat

from RimeBK import RimeBK
from RimeEBK import RimeEBK
from RimeJonesReduce import RimeJonesReduce
#from RimeChiSquared import RimeChiSquared
#from RimeChiSquaredReduce import RimeChiSquaredReduce

from MeasurementSetSharedData import MeasurementSetSharedData

def get_biro_pipeline(msfile, nsrc, device=None):
	"""
	get_biro_pipeline(msfile, nsrc, device=None)

	Returns a pipeline and shared data tuple defining a pipeline
	suitable for BIRO.

	Parameters
	----------
	msfile : string
		Name of the measurement set file.
	nsrc : number
		Number of point sources.
	device - PyCUDA device.
		The CUDA device to execute on If left blank, the default device
		will be selected.

	Returns
	-------
	A (pipeline, shared_data) tuple
	"""
	if device is None:
		import pycuda.autoinit
		device=pycuda.autoinit.device

	# Create a shared data object from the Measurement Set file
	sd = MeasurementSetSharedData(msfile, nsrc=nsrc, dtype=np.float32,
		device=device)
	# Create a pipeline consisting of an EBK kernel, followed by a reduction,
	# a chi squared difference between the Bayesian Model and the Visibilities
	# and a further reduction to produce the Chi Squared Value
	pipeline = Pipeline([
		RimeEBKFloat(),
		RimeJonesReduceFloat(),
		RimeChiSquaredFloat(),
		RimeChiSquaredReduceFloat()])

	return pipeline, sd