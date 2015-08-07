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

import montblanc.impl.biro.v4.gpu.RimeBSqrt

class RimeBSqrt(montblanc.impl.biro.v4.gpu.RimeBSqrt.RimeBSqrt):
    def __init__(self):
        super(RimeBSqrt, self).__init__()
    def initialise(self, solver, stream=None):
        super(RimeBSqrt, self).initialise(solver,stream)
    def shutdown(self, solver, stream=None):
        super(RimeBSqrt, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeBSqrt, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeBSqrt, self).pre_execution(solver,stream)

    def execute(self, solver, stream=None):
        slvr = solver

        self.kernel(slvr.stokes_gpu, slvr.alpha_gpu,
            slvr.frequency_gpu, slvr.B_sqrt_gpu,
            slvr.ref_freq,
            stream=stream, **self.launch_params)