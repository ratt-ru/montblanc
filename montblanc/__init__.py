import inspect
import json
import logging
import logging.config
import numpy as np
import os

# Import ourself. How is this... I don't even...
# Hooray for python
import montblanc

def get_montblanc_path():
	""" Return the current path in which montblanc is installed """
	return os.path.dirname(inspect.getfile(montblanc))

def get_source_path():
	return os.path.join(get_montblanc_path(), 'src')

def get_bk_pipeline(msfile, npsrc, ngsrc, **kwargs):
	"""
	get_bk_pipeline(msfile, npsrc, device=None)

	Returns a pipeline composed of simple brightness and phase terms.

	Parameters
	----------
	msfile : string
		Name of the measurement set file.
	npsrc : number
		Number of point sources.
	ngsrc : number
		Number of gaussian sources.
	device - PyCUDA device.
		The CUDA device to execute on If left blank, the default device
		will be selected.

	Returns
	-------
	A (pipeline, shared_data) tuple
	"""

	import montblanc.factory

	sd = montblanc.factory.get_biro_shared_data(sd_type='ms', msfile=msfile,
		npsrc=npsrc, ngsrc=ngsrc, dtype=np.float32, **kwargs)

	pipeline = montblanc.factory.get_bk_pipeline()

	return pipeline, sd

def get_biro_pipeline(msfile, npsrc, ngsrc, dtype=np.float32, version='v1', **kwargs):
	"""
	get_biro_pipeline(msfile, npsrc, ngsrc, dtype=np.float32, **kwargs)

	Returns a pipeline and shared data tuple defining a pipeline
	suitable for BIRO.

	Parameters
	----------
	msfile : string
		Name of the measurement set file.
	npsrc : number
		Number of point sources.
	ngsrc : number
		Number of gaussian sources.
	dtype : The floating point data type.
		Should be np.float32 or np.float64.
	version : string
		Should be either 'v1' or 'v2'

	Keyword Arguments
	-----------------
	init_weights : string
		Indicates how the weight vector should be initialised from the Measurementset.
		None - Don't initialise the weight vector.
		'sigma' - Initialise from 'SIGMA_SPECTRUM' if present, else 'SIGMA'
		'weight' - Initialise from 'WEIGHT_SPECTRUM' if present, else 'WEIGHT'
	weight_vector : boolean
		True if the chi squared should be computed using a weighting for each value.
		False if it should be computed with a single sigma squared value.
	store_cpu : boolean
		True if copies of the numpy arrays should be stored on the shared data object
		when using the shared data object's transfer_* methods. Otherwise False.
	device - PyCUDA device.
		The CUDA device to execute on If left blank, the default device
		will be selected.

	Returns
	-------
	A (pipeline, shared_data) tuple
	"""

	import montblanc.factory

	sd = montblanc.factory.get_biro_shared_data(sd_type='ms', msfile=msfile,
		npsrc=npsrc, ngsrc=ngsrc, dtype=dtype, version=version, **kwargs)

	pipeline = montblanc.factory.get_biro_pipeline(
		npsrc=npsrc, ngsrc=ngsrc, version=version, **kwargs)

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
