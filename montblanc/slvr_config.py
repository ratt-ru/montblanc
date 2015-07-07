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

from src_types import default_src_cfg
import montblanc.slvr_config_options as Options

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

        self.check_key_values(Options.SRC_CFG, 'Source Configuration Dictionary')
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

    def set_defaults(self, src_cfg=None):
        # Configure sources
        if src_cfg is not None:
            self.setdefault(Options.SRC_CFG, default_src_cfg())
        else:
            self.setdefault(Options.SRC_CFG, src_cfg)

        # Configure the source sizes
        self[Options.SRC_CFG] = src_cfg

        # Configure visibility problem sizeuse the 
        self[Options.NTIME] = Options.DEFAULT_NTIME
        self[Options.NA] = Options.DEFAULT_NA
        self[Options.NCHAN] = Options.DEFAULT_NCHAN

        # Set the data source
        self[Options.DATA_SOURCE] = Options.DEFAULT_DATA_SOURCE 

        # Set the data source
        self[Options.DATA_ORDER] = Options.DEFAULT_DATA_ORDER 

        # Set the data type
        self[Options.DTYPE] = Options.DEFAULT_DTYPE

        # Should we store CPU arrays?
        self[Options.STORE_CPU] = Options.DEFAULT_STORE_CPU