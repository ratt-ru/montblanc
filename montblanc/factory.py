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

def rime_solver(slvr_cfg):
    """ Factory function that produces a BIRO solver """

    # Verify the configuration
    slvr_cfg.verify()

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
    elif data_source == Options.DATA_SOURCE_DEFAULTS:
        return BiroSolver(slvr_cfg)
    else:
        raise Exception, 'Invalid type %s' % sd_type
