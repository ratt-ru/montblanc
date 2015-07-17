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

    E_BEAM_CONFIG = 'E_beam_config'
    E_BEAM_CONFIG_DESCRIPTION = (
        'A dictionary used to configure the E Beam. ',
        'e.g. %s : { \'%s\': %s, \'%s\': %s, \'%s\': %s }' % (
            E_BEAM_CONFIG,
            E_BEAM_WIDTH, DEFAULT_E_BEAM_WIDTH,
            E_BEAM_HEIGHT, DEFAULT_E_BEAM_HEIGHT,
            E_BEAM_DEPTH, DEFAULT_E_BEAM_DEPTH))


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
    INIT_WEIGHT_NONE = None
    INIT_WEIGHT_SIGMA = 'sigma'
    INIT_WEIGHT_WEIGHT = 'weight'
    DEFAULT_INIT_WEIGHT = INIT_WEIGHT_NONE 
    VALID_INIT_WEIGHTS = [INIT_WEIGHT_NONE, INIT_WEIGHT_SIGMA, INIT_WEIGHT_WEIGHT]
    INIT_WEIGHT_DESCRIPTION = (
        "Governs how the weight vector is initialised from a Measurement Set.",
        "If None, uninitialised.",
        "If ''%s'', initialised from the SIGMA column." % INIT_WEIGHT_SIGMA,
        "If ''%s'', initialised from the WEIGHT column." % INIT_WEIGHT_WEIGHT)

    #
    VERSION = 'version'
    VERSION_ONE = 'v1'
    VERSION_TWO = 'v2'
    VERSION_THREE = 'v3'
    VERSION_FOUR = 'v4'
    VERSION_FIVE = 'v5'
    DEFAULT_VERSION = VERSION_FOUR
    VALID_VERSIONS = [VERSION_ONE, VERSION_TWO, VERSION_THREE, VERSION_FOUR]
    VERSION_DESCRIPTION = 'BIRO Version'

    descriptions = {
        WEIGHT_VECTOR: {
            Options.DESCRIPTION: WEIGHT_VECTOR,
            Options.VALID: VALID_WEIGHT_VECTOR,
            Options.DEFAULT: DEFAULT_WEIGHT_VECTOR,
        },

        INIT_WEIGHTS: {
            Options.DESCRIPTION: INIT_WEIGHT_DESCRIPTION,
            Options.VALID: VALID_INIT_WEIGHTS,
            Options.DEFAULT: DEFAULT_INIT_WEIGHT,
        },

        VERSION: {
            Options.DESCRIPTION: VERSION_DESCRIPTION,
            Options.VALID: VALID_VERSIONS,
            Options.DEFAULT: DEFAULT_VERSION,
        },
    }


class BiroSolverConfiguration(SolverConfiguration):
    """
    Object extending the basic Solver Configuration
    with options pertinent to BIRO
    """

    def __init__(self, **kwargs):
        super(BiroSolverConfiguration, self).__init__(**kwargs)

    """
    def __init__(self, mapping, **kwargs):
        super(BiroSolverConfiguration,self).__init__(mapping, **kwargs)

    def __init__(self, iterable, **kwargs):
        super(BiroSolverConfiguration,self).__init__(iterable, **kwargs)
    """


    def verify(self):
        """
        Verify that required parts of the solver configuration
        are present.
        """

        # Do base class checks
        super(BiroSolverConfiguration,self).verify()

        Options = BiroSolverConfigurationOptions

        self.check_key_values(Options.WEIGHT_VECTOR,
            Options.WEIGHT_VECTOR_DESCRIPTION,
            Options.VALID_WEIGHT_VECTOR)

        self.check_key_values(Options.INIT_WEIGHTS,
            Options.INIT_WEIGHT_DESCRIPTION,
            Options.VALID_INIT_WEIGHTS)

        self.check_key_values(Options.VERSION,
            Options.VERSION_DESCRIPTION,
            Options.VALID_VERSIONS)

    def set_defaults(self):
        # Do base class sets
        super(BiroSolverConfiguration,self).set_defaults()

        Options = BiroSolverConfigurationOptions

        # Should we use the weight vector in our calculations?
        self.setdefault(Options.WEIGHT_VECTOR, Options.DEFAULT_WEIGHT_VECTOR)

        # Weight Initialisation scheme
        self.setdefault(Options.INIT_WEIGHTS, Options.DEFAULT_INIT_WEIGHT)

        # Version
        self.setdefault(Options.VERSION, Options.DEFAULT_VERSION)

        # Beam Dimensions
        self.setdefault(Options.E_BEAM_WIDTH, Options.DEFAULT_E_BEAM_WIDTH)
        self.setdefault(Options.E_BEAM_HEIGHT, Options.DEFAULT_E_BEAM_HEIGHT)
        self.setdefault(Options.E_BEAM_DEPTH, Options.DEFAULT_E_BEAM_DEPTH)