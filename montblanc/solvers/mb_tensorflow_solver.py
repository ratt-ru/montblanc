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
import tensorflow as tf

import hypercube as hc
from hypercube.array_factory import (
    create_local_arrays_on_cube,
    create_local_arrays,
    generic_stitch)

from rime_solver import RIMESolver

class MontblancTensorflowSolver(RIMESolver):
    def __init__(self, slvr_cfg):
        super(MontblancTensorflowSolver, self).__init__(slvr_cfg=slvr_cfg)

    def create_default_feed_dict(self):
        # Get our data source
        data_source = self._slvr_cfg[Options.DATA_SOURCE]

        if data_source == Options.DATA_SOURCE_EMPTY:
            return {}

        # Zero by default if we get nothing
        l = lambda s, a: tf.zeros(shape=a.shape, dtype=a.dtype)

        return { n: a.get(data_source, l)(self, a)
            for n, a
            in self.arrays(reify=True).iteritems()
            if not a.temporary
        }
