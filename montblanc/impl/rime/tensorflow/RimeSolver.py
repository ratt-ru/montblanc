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

import montblanc
import montblanc.util as mbu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

from montblanc.impl.rime.v4.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.rime.v4.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.rime.v4.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.rime.v4.gpu.RimeSumCoherencies import RimeSumCoherencies

class RimeSolver(MontblancTensorflowSolver):
    """ RIME Solver Implementation """

    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        super(RimeSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube nu depth')

        # Monkey patch these functions onto the object
        from montblanc.impl.rime.tensorflow.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)
   
        from montblanc.impl.rime.tensorflow.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)
        feed_dict = self.create_default_feed_dict()

        import tensorflow as tf

        expr = feed_dict.values()

        with tf.Session() as S:
            S.run(tf.initialize_all_variables())
            results = [v.flatten()[0:10] for v in S.run(expr)]

            for k, r in zip(feed_dict.iterkeys(), results):
                print k, r
