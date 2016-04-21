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

import montblanc.impl.biro.v4.gpu.RimeEKBSqrt

class RimeEKBSqrt(montblanc.impl.biro.v4.gpu.RimeEKBSqrt.RimeEKBSqrt):
    def __init__(self):
        super(RimeEKBSqrt, self).__init__()
    def initialise(self, solver, stream=None):
        super(RimeEKBSqrt, self).initialise(solver,stream)
    def shutdown(self, solver, stream=None):
        super(RimeEKBSqrt, self).shutdown(solver,stream)
    def pre_execution(self, solver, stream=None):
        super(RimeEKBSqrt, self).pre_execution(solver,stream)
    def post_execution(self, solver, stream=None):
        super(RimeEKBSqrt, self).post_execution(solver,stream)

    def execute(self, solver, stream=None):
        self.kernel(solver.uvw, solver.lm, solver.frequency,
            solver.B_sqrt, solver.jones,
            stream=stream, **self.launch_params)