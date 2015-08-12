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

import montblanc.util as mbu

from montblanc.BaseSolver import BaseSolver

from montblanc.impl.biro.v5.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v5.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v5.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v5.gpu.RimeSumCoherencies import RimeSumCoherencies

from montblanc.impl.biro.v4.BiroSolver import BiroSolver as BiroSolverV4

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

class BiroSolver(BaseSolver):
    """ BIRO Solver Implementation """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """

        super(BiroSolver, self).__init__(slvr_cfg)

        # Configure the dimensions of the beam cube
        self.beam_lw = self.slvr_cfg[Options.E_BEAM_WIDTH]
        self.beam_mh = self.slvr_cfg[Options.E_BEAM_HEIGHT]
        self.beam_nud = self.slvr_cfg[Options.E_BEAM_DEPTH]

        wv = slvr_cfg[Options.WEIGHT_VECTOR]

        self.rime_e_beam = RimeEBeam()
        self.rime_b_sqrt = RimeBSqrt()
        self.rime_ekb_sqrt = RimeEKBSqrt()
        self.rime_sum = RimeSumCoherencies(weight_vector=wv)

    def twiddle_src_dims(self, nsrc):
        """
        Sets self.nsrc = nsrc
        and all other source numbers to 0
        """
        self.nsrc = nsrc
        for nr_var in mbu.source_nr_vars():
            setattr(self, nr_var, 0)

    def get_properties(self):
        # Obtain base solver property dictionary
        # and add the beam cube dimensions to it
        D = super(BiroSolver, self).get_properties()

        D.update({
            'beam_lw' : self.beam_lw,
            'beam_mh' : self.beam_mh,
            'beam_nud' : self.beam_nud
        })

        return D

    def initialise(self):
        with self.context:
            self.rime_e_beam.initialise(self)
            self.rime_b_sqrt.initialise(self)
            self.rime_ekb_sqrt.initialise(self)
            self.rime_sum.initialise(self)

    def shutdown(self):
        with self.context:
            self.rime_e_beam.shutdown(self)
            self.rime_b_sqrt.shutdown(self)
            self.rime_ekb_sqrt.shutdown(self)
            self.rime_sum.shutdown(self)

    # Take these methods from the v4 BiroSolver
    get_default_base_ant_pairs = \
        BiroSolverV4.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BiroSolverV4.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BiroSolverV4.__dict__['get_ap_idx']
