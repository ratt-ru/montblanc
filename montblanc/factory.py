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

import os

import numpy as np

import pycuda.driver as cuda

import montblanc
import montblanc.util as mbu

from montblanc.config import (RimeSolverConfig as Options)

from montblanc.pipeline import Pipeline

__MB_DEVICE_ENV_VAR = 'MONTBLANC_CUDA_DEVICES'

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

        nr_devices = cuda.Device.count()

        if nr_devices == 0:
            raise RuntimeError('Montblanc found no CUDA devices!')

        device_str = os.getenv(__MB_DEVICE_ENV_VAR, '')
        device_str_list = device_str.split()

        # If some valid device string list exists,
        # iterate through it, marking valid devices
        if len(device_str_list) > 0:
            valid_devices = [False for d in range(nr_devices)]

            for i, s in enumerate(device_str_list):
                try:
                    device_nr = int(s)
                    valid_devices[device_nr] = True
                except (ValueError, IndexError) as e:
                    montblanc.log.warn(('The environment variable '
                        '"%s=%s", contains an invalid device number '
                        '"%s" at position %d.') % (__MB_DEVICE_ENV_VAR,
                            device_str, s, i))

            # Are there any valid devices?
            if not np.any(valid_devices):
                montblanc.log.warn(('The environment variable '
                    '"%s=%s", contains no valid device number''s. '
                    'All devices will be selected.') % (
                        __MB_DEVICE_ENV_VAR, device_str))

                valid_devices = [True for d in range(nr_devices)]
    
        # Otherwise, assume every device is valid
        else:
            valid_devices = [True for d in range(nr_devices)]

        # Create the valid devices
        try:
            __devices = [cuda.Device(d)
                for i, d in enumerate(range(nr_devices))
                if valid_devices[i]]
        except:
            raise RuntimeError('Montblanc was unable '
                'to create PyCUDA device objects: %s' % repr(e))

        # Log which devices will be used
        montblanc.log.info('Montblanc will use the following devices:')

        for i, d in enumerate(__devices):
            montblanc.log.info('{p}{d}: {n}'.format(
                p=' '*4, d=i, n=d.name()))

        # Create contexts for each device
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

def get_rime_solver(slvr_cfg):
    """ Get a basic solver object """

    # Get the default cuda context if none is provided
    if slvr_cfg.get(Options.CONTEXT, None) is None:
        slvr_cfg[Options.CONTEXT]=get_default_context()

    from montblanc.solvers.rime_solver import RIMESolver

    return RIMESolver(slvr_cfg=slvr_cfg)

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

    if version in [Options.VERSION_TWO]:
        from montblanc.impl.rime.v2.loaders import MeasurementSetLoader
    elif version == Options.VERSION_FOUR:
        from montblanc.impl.rime.v4.loaders import MeasurementSetLoader
    elif version == Options.VERSION_FIVE:
        from montblanc.impl.rime.v5.loaders import MeasurementSetLoader
    else:
        raise ValueError('Incorrect version %s' % version)

    msfile = slvr_cfg.get(Options.MS_FILE)
    autocor = slvr_cfg.get(Options.AUTO_CORRELATIONS)

    with MeasurementSetLoader(msfile, auto_correlations=autocor) as loader:
        ntime, na, nchan = loader.get_dims(auto_correlations=autocor)
        slvr_cfg[Options.NTIME] = ntime 
        slvr_cfg[Options.NA] = na 
        slvr_cfg[Options.NCHAN] = nchan 
        slvr = slvr_class_type(slvr_cfg)
        loader.load(slvr, slvr_cfg)
        return slvr

def rime_solver(slvr_cfg):
    """ Factory function that produces a BIRO solver """

    # Set the default cuda context if none is provided
    if slvr_cfg.get(Options.CONTEXT, None) is None:
        slvr_cfg[Options.CONTEXT] = get_default_context()

    data_source = slvr_cfg.get(Options.DATA_SOURCE, Options.DATA_SOURCE_MS)
    version = slvr_cfg.get(Options.VERSION, Options.DEFAULT_VERSION)

    # Figure out which version of BIRO solver we're dealing with.
    if version == Options.VERSION_TWO:
        from montblanc.impl.rime.v2.RimeSolver import RimeSolver
    elif version == Options.VERSION_FOUR:
        from montblanc.impl.rime.v4.RimeSolver import RimeSolver
    elif version == Options.VERSION_FIVE:
        from montblanc.impl.rime.v5.CompositeRimeSolver \
        import CompositeRimeSolver as RimeSolver
        slvr_cfg[Options.CONTEXT] = __contexts
    else:
        raise ValueError('Invalid version %s' % version)

    if data_source == Options.DATA_SOURCE_MS:
        return create_rime_solver_from_ms(RimeSolver, slvr_cfg)
    elif data_source == Options.DATA_SOURCE_TEST:
        return RimeSolver(slvr_cfg)
    elif data_source == Options.DATA_SOURCE_DEFAULT:
        return RimeSolver(slvr_cfg)
    elif data_source == Options.DATA_SOURCE_EMPTY:
        return RimeSolver(slvr_cfg)
    else:
        raise ValueError('Invalid data source %s' % data_source)
