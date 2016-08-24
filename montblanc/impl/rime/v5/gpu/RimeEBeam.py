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

import pycuda.driver as cuda

import montblanc.impl.rime.v4.gpu.RimeEBeam

class RimeEBeam(montblanc.impl.rime.v4.gpu.RimeEBeam.RimeEBeam):
    def __init__(self):
        super(RimeEBeam, self).__init__()
    def initialise(self, solver, stream=None):
        super(RimeEBeam, self).initialise(solver,stream)
    def shutdown(self, solver, stream=None):
        super(RimeEBeam, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeEBeam, self).pre_execution(solver,stream)

        if stream is not None:
            cuda.memcpy_htod_async(
                self.rime_const_data[0],
                solver.const_data().ndary(),
                stream=stream)
        else:
            cuda.memcpy_htod(
                self.rime_const_data[0],
                solver.const_data().ndary())

    def post_execution(self, solver, stream=None):
        super(RimeEBeam, self).post_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        self.kernel(slvr.lm, slvr.parallactic_angles,
            slvr.point_errors, slvr.antenna_scaling, slvr.frequency,
            slvr.E_beam, slvr.jones,
            slvr.beam_ll, slvr.beam_lm, slvr.beam_lfreq,
            slvr.beam_ul, slvr.beam_um, slvr.beam_ufreq,
            stream=stream, **self.launch_params)