import numpy as np
import os
import inspect

# Import ourself. How is this... I don't even...
# Hooray for python
import montblanc

from montblanc.node import Node, NullNode
from montblanc.pipeline import Pipeline
from montblanc.BaseSharedData import GPUSharedData

from montblanc.RimeBKFloat import RimeBKFloat
from montblanc.RimeEBKFloat import RimeEBKFloat
from montblanc.RimeJonesReduce import RimeJonesReduceFloat
from montblanc.RimeChiSquaredFloat import RimeChiSquaredFloat
from montblanc.RimeChiSquaredReduceFloat import RimeChiSquaredReduceFloat

from montblanc.RimeBK import RimeBK
from montblanc.RimeEBK import RimeEBK
from montblanc.RimeJonesReduce import RimeJonesReduce
#from montblanc.RimeChiSquared import RimeChiSquared
#from montblanc.RimeChiSquaredReduce import RimeChiSquaredReduce

from montblanc.MeasurementSetSharedData import MeasurementSetSharedData

def get_montblanc_path():
	""" Return the current path in which montblanc is installed """
	return os.path.dirname(inspect.getfile(montblanc))

def get_bk_pipeline(msfile, nsrc, device=None):
	"""
	get_bk_pipeline(msfile, nsrc, device=None)

	Returns a pipeline composed of simple brightness and phase terms.

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
	# Create a pipeline consisting of an BK kernel, followed by a reduction.
	pipeline = Pipeline([
		RimeBKFloat(),
		RimeJonesReduceFloat()])

	return pipeline, sd

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

def default_pipeline_options():
	return {
		'verbosity' : 0
	}
