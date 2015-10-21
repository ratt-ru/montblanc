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

from montblanc.slvr_config import (SolverConfiguration,
    SolverConfigurationOptions as Options)

class BiroSolverConfigurationOptions(Options):
    E_BEAM_WIDTH = 'beam_lw'
    DEFAULT_E_BEAM_WIDTH = 50
    E_BEAM_WIDTH_DESCRIPTION = (
        'Width of the E Beam cube. ',
        'Governs the level of discretisation of '
        'the l dimension.')

    E_BEAM_HEIGHT = 'beam_mh'
    DEFAULT_E_BEAM_HEIGHT = 50
    E_BEAM_HEIGHT_DESCRIPTION = (
        'Height of the E Beam cube. ',
        'Governs the level of discretisation of '
        'the m dimension.')

    E_BEAM_DEPTH = 'beam_nud'
    DEFAULT_E_BEAM_DEPTH = 50
    E_BEAM_DEPTH_DESCRIPTION = (
        'Depth of the E Beam cube. ',
        'Governs the level of discretisation of '
        'the nu (frequency) dimension.')

    # Should a weight vector (sigma) be used to
    # when calculating the chi-squared values?
    WEIGHT_VECTOR = 'weight_vector'
    DEFAULT_WEIGHT_VECTOR = False
    VALID_WEIGHT_VECTOR = [True, False]
    WEIGHT_VECTOR_DESCRIPTION = (
        "If True, chi-squared terms are weighted with a vectorised sigma.",
        "If False, chi-squared terms are weighted with a single scalar sigma.")

    # weight vector initialisation keyword and valid values
    # This options determines whether
    INIT_WEIGHTS = 'init_weights'
    INIT_WEIGHTS_NONE = None
    INIT_WEIGHTS_SIGMA = 'sigma'
    INIT_WEIGHTS_WEIGHT = 'weight'
    DEFAULT_INIT_WEIGHTS = INIT_WEIGHTS_NONE 
    VALID_INIT_WEIGHTS = [INIT_WEIGHTS_NONE, INIT_WEIGHTS_SIGMA, INIT_WEIGHTS_WEIGHT]
    INIT_WEIGHTS_DESCRIPTION = (
        "Governs how the weight vector is initialised from a Measurement Set.",
        "If None, uninitialised.",
        "If ''%s'', initialised from the SIGMA column." % INIT_WEIGHTS_SIGMA,
        "If ''%s'', initialised from the WEIGHT column." % INIT_WEIGHTS_WEIGHT)

    #
    VERSION = 'version'
    VERSION_ONE = 'v1'
    VERSION_TWO = 'v2'
    VERSION_THREE = 'v3'
    VERSION_FOUR = 'v4'
    VERSION_FIVE = 'v5'
    VERSION_SIX = 'v6'
    DEFAULT_VERSION = VERSION_FOUR
    VALID_VERSIONS = [VERSION_ONE, VERSION_TWO, VERSION_THREE,
        VERSION_FOUR, VERSION_FIVE, VERSION_SIX]
    VERSION_DESCRIPTION = 'BIRO Version'

    descriptions = {
        WEIGHT_VECTOR: {
            Options.DESCRIPTION: WEIGHT_VECTOR,
            Options.VALID: VALID_WEIGHT_VECTOR,
            Options.DEFAULT: DEFAULT_WEIGHT_VECTOR,
            Options.REQUIRED: True
        },

        INIT_WEIGHTS: {
            Options.DESCRIPTION: INIT_WEIGHTS_DESCRIPTION,
            Options.VALID: VALID_INIT_WEIGHTS,
            Options.DEFAULT: DEFAULT_INIT_WEIGHTS,
            Options.REQUIRED: True
        },

        VERSION: {
            Options.DESCRIPTION: VERSION_DESCRIPTION,
            Options.VALID: VALID_VERSIONS,
            Options.DEFAULT: DEFAULT_VERSION,
            Options.REQUIRED: True
        },

        E_BEAM_WIDTH: {
            Options.DESCRIPTION: E_BEAM_WIDTH_DESCRIPTION,
            Options.DEFAULT: DEFAULT_E_BEAM_WIDTH,
            Options.REQUIRED: True
        },

        E_BEAM_HEIGHT: {
            Options.DESCRIPTION: E_BEAM_HEIGHT_DESCRIPTION,
            Options.DEFAULT: DEFAULT_E_BEAM_HEIGHT,
            Options.REQUIRED: True
        },

        E_BEAM_DEPTH: {
            Options.DESCRIPTION: E_BEAM_DEPTH_DESCRIPTION,
            Options.DEFAULT: DEFAULT_E_BEAM_DEPTH,
            Options.REQUIRED: True
        },
    }


class BiroSolverConfiguration(SolverConfiguration):
    """
    Object extending the basic Solver Configuration
    with options pertinent to BIRO
    """

    def __init__(self, *args, **kwargs):
        super(BiroSolverConfiguration,self).__init__(*args, **kwargs)
        
    def verify(self, descriptions=None):
        """
        Verify that required parts of the solver configuration
        are present.
        """

        if descriptions is None:
            descriptions = BiroSolverConfigurationOptions.descriptions

        # Do base class checks
        super(BiroSolverConfiguration,self).verify()
        # Now check our class
        super(BiroSolverConfiguration,self).verify(descriptions)

    def set_defaults(self, descriptions=None):
        if descriptions is None:
            descriptions = BiroSolverConfigurationOptions.descriptions

        # Do base class sets
        super(BiroSolverConfiguration,self).set_defaults()
        # Now set our class defaults
        super(BiroSolverConfiguration,self).set_defaults(descriptions)

