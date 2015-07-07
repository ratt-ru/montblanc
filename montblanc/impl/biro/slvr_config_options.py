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
VERSION_ONE = 'v1'
VERSION_TWO = 'v2'
VERSION_THREE = 'v3'
VERSION_FOUR = 'v4'
DEFAULT_VERSION = VERSION_FOUR
VALID_VERSIONS = [VERSION_ONE, VERSION_TWO, VERSION_THREE, VERSION_FOUR]
VERSION_DESCRIPTION = 'BIRO Version'
