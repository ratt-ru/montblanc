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

SRC_CFG = 'src_cfg'

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
DTYPE_FLOAT = np.float32
DTYPE_DOUBLE = np.float64
DEFAULT_DTYPE = DTYPE_DOUBLE
VALID_DTYPES = [DTYPE_FLOAT, DTYPE_DOUBLE]
DTYPE_DESCRIPTION = 'Data type, either a NumPy float or double'

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