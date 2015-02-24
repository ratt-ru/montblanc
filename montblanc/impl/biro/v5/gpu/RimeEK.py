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

import pycuda.driver
import pycuda.tools

import montblanc.impl.biro.v4.gpu.RimeEK

class RimeEK(montblanc.impl.biro.v4.gpu.RimeEK.RimeEK):
    def __init__(self):
        super(RimeEK, self).__init__()
    def initialise(self, solver, stream=None):
        super(RimeEK, self).initialise(solver,stream)
        self.start = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)
        self.end = pycuda.driver.Event(pycuda.driver.event_flags.DISABLE_TIMING)
    def shutdown(self, solver, stream=None):
        super(RimeEK, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeEK, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeEK, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        self.start.record(stream=stream)

        self.kernel(slvr.uvw_gpu, slvr.lm_gpu, slvr.brightness_gpu,
            slvr.wavelength_gpu, slvr.point_errors_gpu, slvr.jones_scalar_gpu,
            slvr.ref_wave, slvr.beam_width, slvr.beam_clip,
            stream=stream, **self.get_kernel_params(slvr))

        self.end.record(stream=stream)
