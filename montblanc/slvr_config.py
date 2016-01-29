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

import montblanc.util as mbu

import numpy as np
import os

class SolverConfigurationOptions(object):
    SOURCES = 'sources'
    DEFAULT_SOURCES = mbu.default_sources()
    SOURCES_DESCRIPTION = (
        'A dictionary defining the number of source types.',
        'Keywords are source types. Values are the number of that source type.')

    # Number of timesteps
    NTIME = 'ntime'
    DEFAULT_NTIME = 10
    NTIME_DESCRIPTION = 'Number of timesteps'

    # Number of antenna
    NA = 'na'
    DEFAULT_NA = 7
    NA_DESCRIPTION = 'Number of antenna'

    # Number of baselines
    NBL = 'nbl'
    NBL_DESCRIPTION = 'Number of baselines'

    # Number of channels
    NCHAN = 'nchan'
    DEFAULT_NCHAN = 16
    NCHAN_DESCRIPTION = 'Number of channels'

    # Number of model parameters for the chi squared gradient
    NPARAMS = 'nparams'
    DEFAULT_NPARAMS = 0
    NPARAMS_DESCRIPTION = 'X2 gradient size' 

    # Are we dealing with floats or doubles?
    DTYPE = 'dtype'
    DTYPE_FLOAT = 'float'
    DTYPE_DOUBLE = 'double'
    DEFAULT_DTYPE = DTYPE_DOUBLE
    VALID_DTYPES = [DTYPE_FLOAT, DTYPE_DOUBLE]
    DTYPE_DESCRIPTION = (
        'Type of floating point precision used to compute the RIME',
        "If '%s', compute the RIME with single-precision" % DTYPE_FLOAT,
        "If '%s', compute the RIME with double-precision" % DTYPE_DOUBLE)

    # Should we handle auto correlations
    AUTO_CORRELATIONS = 'auto_correlations'
    DEFAULT_AUTO_CORRELATIONS = False
    VALID_AUTO_CORRELATIONS = [True, False]
    AUTO_CORRELATIONS_DESCRIPTION = 'Should auto-correlations be taken into account'

    # Data Source. Defaults/A MeasurementSet/Random Test data
    DATA_SOURCE = 'data_source'
    DATA_SOURCE_DEFAULTS = 'defaults'
    DATA_SOURCE_MS = 'ms'
    DATA_SOURCE_TEST = 'test'
    DEFAULT_DATA_SOURCE = DATA_SOURCE_MS
    VALID_DATA_SOURCES = [DATA_SOURCE_DEFAULTS, DATA_SOURCE_MS, DATA_SOURCE_TEST]
    DATA_SOURCE_DESCRIPTION = (
        "The data source for initialising data arrays.",
        "If '%s', data is initialised with defaults." % DATA_SOURCE_DEFAULTS,
        ("If '%s', some data will be read from a MeasurementSet. "
        "otherwise defaults will be used.") % DATA_SOURCE_MS,
        "If '%s' filled with random test data." % DATA_SOURCE_TEST)

    # MeasurementSet file
    MS_FILE = 'msfile'
    MS_FILE_DESCRIPTION = 'MeasurementSet file'

    DATA_ORDER = 'data_order'
    DATA_ORDER_CASA = 'casa'
    DATA_ORDER_OTHER = 'other'
    DEFAULT_DATA_ORDER = DATA_ORDER_CASA
    VALID_DATA_ORDER = [DATA_ORDER_CASA, DATA_ORDER_OTHER]
    DATA_ORDER_DESCRIPTION = (
        "MeasurementSet data ordering",
        "If 'casa' - Assume CASA's default ordering of time x baseline.",
        "If 'other' - Assume baseline x time ordering")

    # Should we store CPU versions when
    # transferring data to the GPU?
    STORE_CPU = 'store_cpu'
    DEFAULT_STORE_CPU = False
    VALID_STORE_CPU = [True, False]
    STORE_CPU_DESCRIPTION = (
        'Governs whether array transfers to the GPU '
        'will be stored in CPU arrays on the solver.')


    CONTEXT = 'context'
    CONTEXT_DESCRIPTION = ('PyCUDA context(s) '
        'available for this solver to use. '
        'Should be of type pycuda.driver.Context. '
        'May be a single context of a list of contexts')

    DESCRIPTION = 'description'
    DEFAULT = 'default'
    VALID = 'valid'
    REQUIRED = 'required'

    descriptions = {
        SOURCES: {
            DESCRIPTION: SOURCES_DESCRIPTION,
            DEFAULT: DEFAULT_SOURCES,
            REQUIRED: True },

        NTIME: {
            DESCRIPTION: NTIME_DESCRIPTION,
            DEFAULT: DEFAULT_NTIME,
            REQUIRED: True },

        NA: {
            DESCRIPTION: NA_DESCRIPTION,
            DEFAULT: DEFAULT_NA,
            REQUIRED: True },

        NBL: {
            DESCRIPTION: NBL_DESCRIPTION,
            REQUIRED: False },

        NCHAN: {
            DESCRIPTION: NCHAN_DESCRIPTION,
            DEFAULT: DEFAULT_NCHAN,
            REQUIRED: True },

        NPARAMS: {
            DESCRIPTION: NPARAMS_DESCRIPTION,
            DEFAULT: DEFAULT_NPARAMS,
            REQUIRED: False },    

        DTYPE: {
            DESCRIPTION: DTYPE_DESCRIPTION,
            DEFAULT: DEFAULT_DTYPE,
            VALID: VALID_DTYPES,
            REQUIRED: True },

        AUTO_CORRELATIONS: {
            DESCRIPTION: AUTO_CORRELATIONS_DESCRIPTION,
            DEFAULT: DEFAULT_AUTO_CORRELATIONS,
            VALID: VALID_AUTO_CORRELATIONS
        },

        DATA_SOURCE: {
            DESCRIPTION: DATA_SOURCE_DESCRIPTION,
            DEFAULT: DEFAULT_DATA_SOURCE,
            VALID: VALID_DATA_SOURCES,
            REQUIRED: True
        },

        MS_FILE: {
            DESCRIPTION:  MS_FILE_DESCRIPTION,
        },

        DATA_ORDER: {
            DESCRIPTION: DATA_ORDER_DESCRIPTION,
            DEFAULT: DEFAULT_DATA_ORDER,
            VALID: VALID_DATA_ORDER,
            REQUIRED: True
        },

        CONTEXT : {
            DESCRIPTION: CONTEXT_DESCRIPTION,
            REQUIRED: True
        },

        STORE_CPU: {
            DESCRIPTION: STORE_CPU_DESCRIPTION,
            DEFAULT: DEFAULT_STORE_CPU,
            VALID: VALID_STORE_CPU
        },
    }

class SolverConfiguration(dict):
    """
    Object defining the Solver Configuration. Inherits from dict.

    Tracks the following quantities defining the problem size
      - timesteps
      - antenna
      - channels
      - sources
      - chi squared gradient size (Optional)

    the data type
      - float
      - double

    the data source for the solver
      - None (data takes defaults)
      - A Measurement Set
      - Random data
    """

    def __init__(self, *args, **kwargs):
        """
        Construct a default solver configuration
        """
        super(SolverConfiguration,self).__init__(*args, **kwargs)
        self.set_defaults()

    def check_key_values(self, key, description=None, valid_values=None):
        if description is None:
            description = "" 

        if key not in self:
            raise KeyError(("Solver configuration is missing "
                "key ''%s''. Description: %s") % (key, description))

        if valid_values is not None and self[key] not in valid_values:
            raise KeyError(("Value ''%s'' for solver configuration "
                "key ''%s'' is invalid. "
                "Valid values are %s.") % (self[key], key, valid_values))

    def verify(self, descriptions=None):
        """
        Verify that required parts of the solver configuration
        are present.
        """

        Options = SolverConfigurationOptions

        if descriptions is None:
            descriptions = Options.descriptions

        for name, info in descriptions.iteritems():
            required = info.get(Options.REQUIRED, False)

            if required and name not in self:
                description = info.get(Options.DESCRIPTION, 'None')

                raise KeyError(("Solver configuration is missing "
                    "key '%s'. Description: %s") % (name, description))

            valid_values = info.get(Options.VALID, None)

            if valid_values is not None and self[name] not in valid_values:
                raise KeyError(("Value '%s for solver configuration "
                    "key '%s' is invalid. "
                    "Valid values are %s.") % (self[name], name, valid_values))

    def set_defaults(self, descriptions=None):
        Options = SolverConfigurationOptions

        if descriptions is None:
            descriptions = Options.descriptions

        for name, info in descriptions.iteritems():
            if Options.DEFAULT in info:
                self.setdefault(name, info.get(Options.DEFAULT))
