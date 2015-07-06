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
import os

from src_types import default_src_cfg

SRC_CFG = 'src_cfg'

# Number of timesteps
NTIME = 'ntime'
DEFAULT_NTIME = 5

# Number of antenna
NA = 'na'
DEFAULT_NA = 5

# Number of channels
NCHAN = 'nchan'
DEFAULT_NCHAN = 5

# Are we dealing with floats or doubles?
DTYPE = 'dtype'
DTYPE_FLOAT = np.float32
DTYPE_DOUBLE = np.float64
DEFAULT_DTYPE = DTYPE_DOUBLE
VALID_DTYPES = [DTYPE_FLOAT, DTYPE_DOUBLE]
DTYPE_DESCRIPTION = 'Data type, either a NumPy float or double'

# Data Source. Nothing/A MeasurementSet/Random Test data
DATA_SOURCE = 'data_source'
DATA_SOURCE_NONE = None
DATA_SOURCE_MS = 'ms'
DATA_SOURCE_TEST = 'test'
DEFAULT_DATA_SOURCE = DATA_SOURCE_NONE
VALID_DATA_SOURCES = [DATA_SOURCE_NONE, DATA_SOURCE_MS, DATA_SOURCE_TEST]
DATA_SOURCE_DESCRIPTION = os.linesep.join((
    'Data source.',
    'If ''None'', data is initialised with defaults.',
    'If ''%s'', some data will be read from a MeasurementSet,' % DATA_SOURCE_MS,
    'or if ''%s'' filled with random test data.' % DATA_SOURCE_TEST))

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

    def __init__(self, src_cfg=None):
        """
        Construct a default solver configuration
        """
        super(SolverConfiguration,self).__init__()
        self.set_defaults(src_cfg)

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

        self.check_key_values(SRC_CFG, 'Source Configuration Dictionary')
        self.check_key_values(NTIME, 'Number of Timesteps')
        self.check_key_values(NA, 'Number of Antenna')
        self.check_key_values(NCHAN, 'Number of Channels')

        self.check_key_values(DTYPE, DTYPE_DESCRIPTION, VALID_DTYPES)

        self.check_key_values(DATA_SOURCE, DATA_SOURCE_DESCRIPTION,
            VALID_DATA_SOURCES)

        self.check_key_values(DATA_ORDER, DATA_ORDER_DESCRIPTION,
            VALID_DATA_ORDER)

        self.check_key_values(STORE_CPU, STORE_CPU_DESCRIPTION,
            VALID_STORE_CPU)

    def set_defaults(self, src_cfg=None):
        # Configure sources
        if src_cfg is not None:
            self.setdefault(SRC_CFG, default_src_cfg())
        else:
            self.setdefault(SRC_CFG, src_cfg)

        # Configure visibility problem sizeuse the 
        self.setdefault(NTIME, DEFAULT_NTIME)
        self.setdefault(NA, DEFAULT_NA)
        self.setdefault(NCHAN, DEFAULT_NCHAN)

        # Configure the source sizes
        self.setdefault(SRC_CFG, src_cfg)

        # Set the data source
        self.setdefault(DATA_SOURCE, DEFAULT_DATA_SOURCE) 

        # Set the data source
        self.setdefault(DATA_ORDER, DEFAULT_DATA_ORDER) 

        # Set the data type
        self.setdefault(DTYPE, DEFAULT_DTYPE)

        # Should we store CPU arrays?
        self.setdefault(STORE_CPU, DEFAULT_STORE_CPU)