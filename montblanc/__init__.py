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

import montblanc.config

from montblanc.tests import test
from montblanc.version import __version__

def get_montblanc_path():
    """ Return the current path in which montblanc is installed """
    import montblanc
    return os.path.dirname(inspect.getfile(montblanc))

def get_source_path():
    return os.path.join(get_montblanc_path(), 'src')

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

def source_types():
    """
    Returns the source types available in montblanc

    >>> montblanc.source_types()
    %s
    """
    return montblanc.src_types.SOURCE_VAR_TYPES.keys()

source_types.__doc__ %= montblanc.src_types.SOURCE_VAR_TYPES.keys()

def sources(**kwargs):
    """
    Keyword arguments
    -----------------
    Keyword argument names should be name of source types
    registered with montblanc. Their values should be
    the number of these sources types.

    Returns
    -------
    A dict containing source type numbers for all
    valid source types (%s).

    >>> %s
    %s

    """    
    return montblanc.src_types.default_sources(**kwargs)

# Substitute docstring variables
sources.__doc__ %= (', '.join(source_types()),
    'montblanc.sources(point=10, gaussian=20)',
    sources(point=10, gaussian=20))

def rime_solver_cfg(**kwargs):
    """
    Produces a SolverConfiguration object, inherited from
    a simple python dict, and containing the options required
    to configure the RIME Solver.

    Keyword arguments
    -----------------
    Any keyword arguments are inserted into the
    returned dict.

    Returns
    -------
    A SolverConfiguration object.

    montblanc.rime_solver_cfg(msfile='WSRT.MS',
        sources=montblanc.source(point=10,gaussian=20))

    Valid configuration options are
    %s
    """

    from montblanc.impl.biro.slvr_config import (BiroSolverConfiguration,
        BiroSolverConfigurationOptions as Options)

    slvr_cfg = BiroSolverConfiguration(**kwargs)

    # Assume a MeasurementSet data source, if none is supplied
    slvr_cfg.setdefault(Options.DATA_SOURCE, Options.DATA_SOURCE_MS)

    return slvr_cfg

rime_solver_cfg.__doc__ %= (montblanc.config.describe_options())

def rime_solver(slvr_cfg):
    """
    rime_solver(slvr_cfg)

    Returns a solver suitable for solving the RIME.

    Parameters
    ----------
    slvr_cfg : BiroSolverConfiguration
            Solver Configuration.

    Returns
    -------
    A solver
    """

    import montblanc.factory

    return montblanc.factory.rime_solver(slvr_cfg)