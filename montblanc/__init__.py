import inspect
import json
import logging
import logging.config
import numpy as np
import os

# Import ourself. How is this... I don't even...
# Hooray for python
import montblanc

from montblanc.node import Node, NullNode
from montblanc.pipeline import Pipeline
from montblanc.BaseSharedData import GPUSharedData

from montblanc.RimeBK import RimeBK
from montblanc.RimeEBK import RimeEBK
from montblanc.RimeJonesReduce import RimeJonesReduce
from montblanc.RimeChiSquared import RimeChiSquared
from montblanc.RimeChiSquaredReduce import RimeChiSquaredReduce

from montblanc.MeasurementSetSharedData import MeasurementSetSharedData

def get_montblanc_path():
	""" Return the current path in which montblanc is installed """
	return os.path.dirname(inspect.getfile(montblanc))

def get_source_path():
	return os.path.join(get_montblanc_path(), 'src')

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
		RimeBK(),
		RimeJonesReduce()])

	return pipeline, sd

def get_biro_pipeline(msfile, nsrc, noise_vector=False, device=None):
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
	noise_vector : boolean
		True if the chi squared should be computed with a noise vector.
		False if it should be computed with a single sigma squared value.
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
		RimeEBK(),
		RimeJonesReduce(),
		RimeChiSquared(),
		RimeChiSquaredReduce(noise_vector=noise_vector)])

	return pipeline, sd

def setup_logging(default_level=logging.INFO,env_key='LOG_CFG'):
    """ Setup logging configuration """

    path = os.path.join(get_montblanc_path(), 'log', 'log.json')
    value = os.getenv(env_key, None)

    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

setup_logging()
log = logging.getLogger('montblanc')

# This solution for constants based on
# http://stackoverflow.com/a/2688086
# Create a property that throws when
# you try and set it
def constant(f):
    def fset(self, value):
        raise SyntaxError, 'Foolish Mortal! You would dare change a universal constant?'
    def fget(self):
        return f()

    return property(fget, fset)

class MontblancConstants(object):
    # The speed of light, in metres
    @constant
    def C():
        return 299792458

# Create a constants object
constants = MontblancConstants()
