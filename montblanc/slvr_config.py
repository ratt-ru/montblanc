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

    # Number of timesteps
    NTIME = 'ntime'
    DEFAULT_NTIME = 5
    NTIME_DESCRIPTION = 'Number of timesteps'

    # Number of antenna
    NA = 'na'
    DEFAULT_NA = 5
    NA_DESCRIPTION = 'Number of antenna'

    # Number of channels
    NCHAN = 'nchan'
    DEFAULT_NCHAN = 5
    NCHAN_DESCRIPTION = 'Number of channels'

    # Are we dealing with floats or doubles?
    DTYPE = 'dtype'
    DTYPE_FLOAT = 'float'
    DTYPE_DOUBLE = 'double'
    DEFAULT_DTYPE = DTYPE_DOUBLE
    VALID_DTYPES = [DTYPE_FLOAT, DTYPE_DOUBLE]
    DTYPE_DESCRIPTION = 'Data type, either a NumPy float or double'

    # Should we handle auto correlations
    AUTO_CORRELATIONS = 'auto_correlations'
    DEFAULT_AUTO_CORRELATIONS = False
    VALID_AUTO_CORRELATIONS = [True, False]

    # Data Source. Nothing/A MeasurementSet/Random Test data
    DATA_SOURCE = 'data_source'
    DATA_SOURCE_BIRO = 'biro'
    DATA_SOURCE_MS = 'ms'
    DATA_SOURCE_TEST = 'test'
    DEFAULT_DATA_SOURCE = DATA_SOURCE_BIRO
    VALID_DATA_SOURCES = [DATA_SOURCE_BIRO, DATA_SOURCE_MS, DATA_SOURCE_TEST]
    DATA_SOURCE_DESCRIPTION = os.linesep.join((
        'Data source.',
        'If ''None'', data is initialised with defaults.',
        'If ''%s'', some data will be read from a MeasurementSet,' % DATA_SOURCE_MS,
        'or if ''%s'' filled with random test data.' % DATA_SOURCE_TEST))

    # MeasurementSet file
    MS_FILE = 'msfile'

    DATA_ORDER = 'data_order'
    DATA_ORDER_CASA = 'casa'
    DATA_ORDER_OTHER = 'other'
    DEFAULT_DATA_ORDER = DATA_ORDER_CASA
    VALID_DATA_ORDER = [DATA_ORDER_CASA, DATA_ORDER_OTHER]
    DATA_ORDER_DESCRIPTION = os.linesep.join((
        'MeasurementSet data ordering: time x baseline or baseline x time. ',
        'casa - Assume CASA''s default ordering of time x baseline. ',
        'other - Assume baseline x time ordering') )

    # Should we store CPU versions when
    # transferring data to the GPU?
    STORE_CPU = 'store_cpu'
    DEFAULT_STORE_CPU = False
    VALID_STORE_CPU = [True, False]
    STORE_CPU_DESCRIPTION = (
        'Governs whether array transfers to the GPU '
        'will be stored in CPU arrays on the solver.')

class SolverConfiguration(dict):
    """
    Object defining the Solver Configuration. Inherits from dict.

    Tracks the following quantities defining the problem size
      - timesteps
      - antenna
      - channels
      - sources

    the data type
      - float
      - double

    the data source for the solver
      - None (data takes defaults)
      - A Measurement Set
      - Random data
    """

    def __init__(self, **kwargs):
        """
        Construct a default solver configuration
        """
        super(SolverConfiguration,self).__init__(**kwargs)
        self.set_defaults()

    """
    def __init__(self, mapping, **kwargs):
        super(SolverConfiguration,self).__init__(mapping, **kwargs)
        self.set_defaults()

    def __init__(self, iterable, **kwargs):
        super(SolverConfiguration,self).__init__(iterable, **kwargs)
        self.set_defaults()
    """


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

    def verify(self):
        """
        Verify that required parts of the solver configuration
        are present.
        """

        Options = SolverConfigurationOptions

        self.check_key_values(Options.SOURCES, 'Source Configuration Dictionary')
        self.check_key_values(Options.NTIME, Options.NTIME_DESCRIPTION)
        self.check_key_values(Options.NA, Options.NA_DESCRIPTION)
        self.check_key_values(Options.NCHAN, Options.NCHAN_DESCRIPTION)

        self.check_key_values(Options.DTYPE,
            Options.DTYPE_DESCRIPTION, Options.VALID_DTYPES)

        self.check_key_values(Options.DATA_SOURCE,
            Options.DATA_SOURCE_DESCRIPTION, Options.VALID_DATA_SOURCES)

        self.check_key_values(Options.DATA_ORDER,
            Options.DATA_ORDER_DESCRIPTION, Options.VALID_DATA_ORDER)

        self.check_key_values(Options.STORE_CPU,
            Options.STORE_CPU_DESCRIPTION, Options.VALID_STORE_CPU)

    def set_defaults(self):
        Options = SolverConfigurationOptions

        # Configure Sources
        self[Options.SOURCES] = mbu.default_sources()

        # Configure visibility problem sizeuse the 
        self[Options.NTIME] = Options.DEFAULT_NTIME
        self[Options.NA] = Options.DEFAULT_NA
        self[Options.NCHAN] = Options.DEFAULT_NCHAN

        # Should
        self[Options.AUTO_CORRELATIONS] = Options.DEFAULT_AUTO_CORRELATIONS

        # Set the data source
        self[Options.DATA_SOURCE] = Options.DEFAULT_DATA_SOURCE 

        # Set the data source
        self[Options.DATA_ORDER] = Options.DEFAULT_DATA_ORDER 

        # Set the data type
        self[Options.DTYPE] = Options.DEFAULT_DTYPE

        # Should we store CPU arrays?
        self[Options.STORE_CPU] = Options.DEFAULT_STORE_CPU