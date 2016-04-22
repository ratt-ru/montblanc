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

from montblanc.slvr_config import SolverConfig

def _init_weights(s):
    """ Handle the None value in VALID_INIT_WEIGHTS """
    if s not in RimeSolverConfig.VALID_INIT_WEIGHTS:
        import configargparse

        raise configargparse.ArgumentTypeError("\'{iw}\'' must be one of {viw}"
            .format(iw=INIT_WEIGHTS, viw=RimeSolverConfig.VALID_INIT_WEIGHTS))

    return s

class RimeSolverConfig(SolverConfig):
    E_BEAM_WIDTH = 'beam_lw'
    DEFAULT_E_BEAM_WIDTH = 50
    E_BEAM_WIDTH_DESCRIPTION = (
        'Width of the E Beam cube. '
        'Governs the level of discretisation of '
        'the l dimension.')

    E_BEAM_HEIGHT = 'beam_mh'
    DEFAULT_E_BEAM_HEIGHT = 50
    E_BEAM_HEIGHT_DESCRIPTION = (
        'Height of the E Beam cube. '
        'Governs the level of discretisation of '
        'the m dimension.')

    E_BEAM_DEPTH = 'beam_nud'
    DEFAULT_E_BEAM_DEPTH = 50
    E_BEAM_DEPTH_DESCRIPTION = (
        'Depth of the E Beam cube. '
        'Governs the level of discretisation of '
        'the nu (frequency) dimension.')

    # Should a weight vector (sigma) be used to
    # when calculating the chi-squared values?
    WEIGHT_VECTOR = 'weight_vector'
    DEFAULT_WEIGHT_VECTOR = False
    VALID_WEIGHT_VECTOR = [True, False]
    WEIGHT_VECTOR_DESCRIPTION = (
        "If True, chi-squared terms are weighted with a vectorised sigma. "
        "If False, chi-squared terms are weighted with a single scalar sigma.")

    # weight vector initialisation keyword and valid values
    # This SolverConfig determines whether
    INIT_WEIGHTS = 'init_weights'
    INIT_WEIGHTS_NONE = None
    INIT_WEIGHTS_SIGMA = 'sigma'
    INIT_WEIGHTS_WEIGHT = 'weight'
    DEFAULT_INIT_WEIGHTS = INIT_WEIGHTS_NONE 
    VALID_INIT_WEIGHTS = [INIT_WEIGHTS_NONE, INIT_WEIGHTS_SIGMA, INIT_WEIGHTS_WEIGHT]
    INIT_WEIGHTS_DESCRIPTION = (
        "Governs how the weight vector is initialised from a Measurement Set. "
        "If None, uninitialised. "
        "If '{s}', initialised from the SIGMA column."
        "If '{w}', initialised from the WEIGHT column.").format(
            s=INIT_WEIGHTS_SIGMA, w=INIT_WEIGHTS_WEIGHT)

    SOURCE_BATCH_SIZE = 'source_batch_size'
    DEFAULT_SOURCE_BATCH_SIZE = 500
    SOURCE_BATCH_SIZE_DESCRIPTION = (
        "Minimum source batch size used when computing the RIME")

    NSOLVERS = 'nsolvers'
    DEFAULT_NSOLVERS = 2
    NSOLVERS_DESCRIPTION = (
        "Number of concurrent GPU solvers per device")

    VISIBILITY_THROTTLE_FACTOR = 'visibility_throttle_factor'
    DEFAULT_VISIBILITY_THROTTLE_FACTOR = 6
    VISIBILITY_THROTTLE_FACTOR_DESCRIPTION = (
        "Maximum number of visibility chunks that may be "
        "enqueued on a solver before throttling is applied.")

    # RIME version
    VERSION = 'version'
    VERSION_ONE = 'v1'
    VERSION_TWO = 'v2'
    VERSION_THREE = 'v3'
    VERSION_FOUR = 'v4'
    VERSION_FIVE = 'v5'
    DEFAULT_VERSION = VERSION_FOUR
    VALID_VERSIONS = [VERSION_TWO, VERSION_FOUR, VERSION_FIVE]
    VERSION_DESCRIPTION = 'RIME Version'

    descriptions = {
        WEIGHT_VECTOR: {
            SolverConfig.DESCRIPTION: WEIGHT_VECTOR_DESCRIPTION,
            SolverConfig.VALID: VALID_WEIGHT_VECTOR,
            SolverConfig.DEFAULT: DEFAULT_WEIGHT_VECTOR,
            SolverConfig.REQUIRED: True
        },

        INIT_WEIGHTS: {
            SolverConfig.DESCRIPTION: INIT_WEIGHTS_DESCRIPTION,
            SolverConfig.VALID: VALID_INIT_WEIGHTS,
            SolverConfig.DEFAULT: DEFAULT_INIT_WEIGHTS,
            SolverConfig.REQUIRED: True
        },

        SOURCE_BATCH_SIZE: {
            SolverConfig.DESCRIPTION: SOURCE_BATCH_SIZE_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_SOURCE_BATCH_SIZE,
            SolverConfig.REQUIRED: True
        },

        VISIBILITY_THROTTLE_FACTOR: {
            SolverConfig.DESCRIPTION: VISIBILITY_THROTTLE_FACTOR_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_VISIBILITY_THROTTLE_FACTOR,
            SolverConfig.REQUIRED: True
        },

        NSOLVERS: {
            SolverConfig.DESCRIPTION: NSOLVERS_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_NSOLVERS,
            SolverConfig.REQUIRED: True
        },

        VERSION: {
            SolverConfig.DESCRIPTION: VERSION_DESCRIPTION,
            SolverConfig.VALID: VALID_VERSIONS,
            SolverConfig.DEFAULT: DEFAULT_VERSION,
            SolverConfig.REQUIRED: True
        },

        E_BEAM_WIDTH: {
            SolverConfig.DESCRIPTION: E_BEAM_WIDTH_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_E_BEAM_WIDTH,
            SolverConfig.REQUIRED: True
        },

        E_BEAM_HEIGHT: {
            SolverConfig.DESCRIPTION: E_BEAM_HEIGHT_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_E_BEAM_HEIGHT,
            SolverConfig.REQUIRED: True
        },

        E_BEAM_DEPTH: {
            SolverConfig.DESCRIPTION: E_BEAM_DEPTH_DESCRIPTION,
            SolverConfig.DEFAULT: DEFAULT_E_BEAM_DEPTH,
            SolverConfig.REQUIRED: True
        },
    }

    def parser(self):
        p = super(RimeSolverConfig, self).parser()

        p.add_argument('--{v}'.format(v=self.E_BEAM_WIDTH),
            required=False,
            type=int,
            help=self.E_BEAM_WIDTH_DESCRIPTION,
            default=self.DEFAULT_E_BEAM_WIDTH)

        p.add_argument('--{v}'.format(v=self.E_BEAM_HEIGHT),
            required=False,
            type=int,
            help=self.E_BEAM_HEIGHT_DESCRIPTION,
            default=self.DEFAULT_E_BEAM_HEIGHT)

        p.add_argument('--{v}'.format(v=self.E_BEAM_DEPTH),
            required=False,
            type=int,
            help=self.E_BEAM_DEPTH_DESCRIPTION,
            default=self.DEFAULT_E_BEAM_DEPTH)

        p.add_argument('--{v}'.format(v=self.WEIGHT_VECTOR),
            required=False,
            type=bool,
            choices=self.VALID_WEIGHT_VECTOR,
            help=self.WEIGHT_VECTOR_DESCRIPTION,
            default=self.DEFAULT_WEIGHT_VECTOR)

        p.add_argument('--{v}'.format(v=self.INIT_WEIGHTS),
            required=False,
            type=_init_weights,
            choices=self.VALID_INIT_WEIGHTS,
            help=self.INIT_WEIGHTS_DESCRIPTION,
            default=self.DEFAULT_INIT_WEIGHTS)

        p.add_argument('--{v}'.format(v=self.SOURCE_BATCH_SIZE),
            required=False,
            type=int,
            help=self.SOURCE_BATCH_SIZE_DESCRIPTION,
            default=self.DEFAULT_SOURCE_BATCH_SIZE)

        p.add_argument('--{v}'.format(v=self.NSOLVERS),
            required=False,
            type=int,
            help=self.NSOLVERS_DESCRIPTION,
            default=self.DEFAULT_NSOLVERS)

        p.add_argument('--{v}'.format(v=self.VISIBILITY_THROTTLE_FACTOR),
            required=False,
            type=int,
            help=self.VISIBILITY_THROTTLE_FACTOR_DESCRIPTION,
            default=self.DEFAULT_VISIBILITY_THROTTLE_FACTOR)

        p.add_argument('--{v}'.format(v=self.VERSION),
            required=False,
            type=str,
            choices=self.VALID_VERSIONS,
            help=self.VERSION_DESCRIPTION,
            default=self.DEFAULT_VERSION)

        return p