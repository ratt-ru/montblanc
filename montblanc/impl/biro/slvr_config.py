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

# 
WEIGHT_VECTOR = 'weight_vector'
DEFAULT_WEIGHT_VECTOR = False
VALID_WEIGHT_VECTOR = [True, False]
WEIGHT_VECTOR_DESCRIPTION = (
    'Governs whether chi-squared terms is weighted with vectorised, '
    'or single scalar sigma value.')

# weight vector initialisation keyword and valid values
# This options determines whether
INIT_WEIGHT = 'init_weight'
INIT_WEIGHT_NONE = None
INIT_WEIGHT_SIGMA = 'sigma'
INIT_WEIGHT_WEIGHT = 'weight'
DEFAULT_INIT_WEIGHT = INIT_WEIGHT_NONE 
VALID_INIT_WEIGHTS = [INIT_WEIGHT_NONE, INIT_WEIGHT_SIGMA, INIT_WEIGHT_WEIGHT]
INIT_WEIGHT_DESCRIPTION = (
    'Governs how the weight vector is initialised from a Measurement Set. '
    'If None, uninitialised. '
    'If ''%s'', initialised from the SIGMA column. '
    'If ''%s'', initialised from the WEIGHT column.')

#
VERSION = 'version'
VERSION_V1 = 'v1'
VERSION_V2 = 'v2'
VERSION_V3 = 'v3'
VERSION_V4 = 'v4'
DEFAULT_VERSION = VERSION_V4
VALID_VERSIONS = [VERSION_V1, VERSION_V2, VERSION_V3, VERSION_V4]
VERSION_DESCRIPTION = 'BIRO Version'

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

        self.check_key_values(WEIGHT_VECTOR,
            WEIGHT_VECTOR_DESCRIPTION,
            VALID_WEIGHT_VECTOR)

        self.check_key_values(INIT_WEIGHT,
            INIT_WEIGHT_DESCRIPTION,
            VALID_INIT_WEIGHTS)

        self.check_key_values(VERSION,
            VERSION_DESCRIPTION,
            VALID_VERSIONS)

    def set_defaults(self, src_cfg=None):
        # Do base class sets
        super(BiroSolverConfiguration,self).set_defaults(src_cfg)

        # Should we use the weight vector in our calculations?
        self[WEIGHT_VECTOR] = DEFAULT_WEIGHT_VECTOR

        # Weight Initialisation scheme
        self[INIT_WEIGHT] = DEFAULT_INIT_WEIGHT

        # Version
        self[VERSION] =  DEFAULT_VERSION