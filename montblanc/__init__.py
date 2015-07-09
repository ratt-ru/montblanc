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

import inspect
import json
import logging
import logging.config
import numpy as np
import os

# Import ourself. How is this... I don't even...
# Hooray for python
import montblanc
import montblanc.util as mbu

from montblanc.tests import test
from montblanc.version import __version__

def get_montblanc_path():
    """ Return the current path in which montblanc is installed """
    return os.path.dirname(inspect.getfile(montblanc))

def get_source_path():
    return os.path.join(get_montblanc_path(), 'src')

def sources(**kwargs):
    """
    Given arguments which are numbers of point types

    e.g. sources(point=10, gaussian=20, fake=30)

    returns a dict defining the number of all valid point types
        { 'point': 10, 'gaussian': 20, 'sersic': 0 }
    """
    return mbu.default_sources(**kwargs)

def biro_solver_cfg(**kwargs):
    """
    Returns a BiroSolverConfiguration object, inherited from
    a simple python dict, and containing the options required
    to configure the Biro Solver.
    """
    from montblanc.impl.biro.slvr_config import (BiroSolverConfiguration,
        BiroSolverConfigurationOptions as Options)

    slvr_cfg = BiroSolverConfiguration(**kwargs)

    # Assume a MeasurementSet data source by default
    slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_MS

    if slvr_cfg[Options.DATA_SOURCE] == Options.DATA_SOURCE_MS and \
        Options.MS_FILE not in slvr_cfg:

        raise KeyError(('%s key is set to %s '
            'in the Solver Configuration, but '
            'no MeasurementSet file has been '
            'specified in the %s key') % (
                Options.DATA_SOURCE,
                Options.DATA_SOURCE_MS,
                Options.MS_FILE))  

    return slvr_cfg

def get_biro_solver(slvr_cfg):
    """
    get_biro_solver(msfile, npsrc, ngsrc, nssrc, dtype=np.float32, **kwargs)

    Returns a solver suitable for solving the BIRO RIME.

    Parameters
    ----------
    msfile : string
            Name of the measurement set file.
    npsrc : number
            Number of point sources.
    ngsrc : number
            Number of gaussian sources.
    nssrc : number
            Number of sersic sources.
    dtype : The floating point data type.
            Should be np.float32 or np.float64.
    version : string
            Should be either 'v1' or 'v2'

    Keyword Arguments
    -----------------
    data_order : string
	    Indicates what is the MeasurementSet data ordering: time x baseline or baseline x time.
	    None - Assume Montblanc's default ordering (time x baseline)
	    'casa' - Assume CASA's default ordering. It matches Montblanc's default ordering, so it can be avoided.
	    'other' - Assume baseline x time ordering
    
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
    context - pycuda.driver.Context.
            The CUDA context to execute on If left blank, the default context
            will be selected.

    Returns
    -------
    A solver
    """

    import montblanc.factory

    return montblanc.factory.get_biro_solver(slvr_cfg)

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
