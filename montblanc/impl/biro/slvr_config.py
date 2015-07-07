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

from montblanc.slvr_config import SolverConfiguration
import montblanc.impl.biro.slvr_config_options as BiroOptions

class BiroSolverConfiguration(SolverConfiguration):
    """
    Object extending the basic Solver Configuration
    with options pertinent to BIRO
    """

    def __init__(self, src_cfg=None):
        super(BiroSolverConfiguration, self).__init__(src_cfg)
        #self.setdefault(src_cfg)

    def verify(self):
        """
        Verify that required parts of the solver configuration
        are present.
        """

        # Do base class checks
        super(BiroSolverConfiguration,self).verify()

        self.check_key_values(BiroOptions.WEIGHT_VECTOR,
            BiroOptions.WEIGHT_VECTOR_DESCRIPTION,
            BiroOptions.VALID_WEIGHT_VECTOR)

        self.check_key_values(BiroOptions.INIT_WEIGHT,
            BiroOptions.INIT_WEIGHT_DESCRIPTION,
            BiroOptions.VALID_INIT_WEIGHTS)

        self.check_key_values(BiroOptions.VERSION,
            BiroOptions.VERSION_DESCRIPTION,
            BiroOptions.VALID_VERSIONS)

    def set_defaults(self, src_cfg=None):
        # Do base class sets
        super(BiroSolverConfiguration,self).set_defaults(src_cfg)

        # Should we use the weight vector in our calculations?
        self[BiroOptions.WEIGHT_VECTOR] = BiroOptions.DEFAULT_WEIGHT_VECTOR

        # Weight Initialisation scheme
        self[BiroOptions.INIT_WEIGHT] = BiroOptions.DEFAULT_INIT_WEIGHT

        # Version
        self[BiroOptions.VERSION] = BiroOptions.DEFAULT_VERSION