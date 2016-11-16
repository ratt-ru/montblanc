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

def load_tf_lib():
    """ Load the tensorflow library """
    import tensorflow as tf
    from tensorflow.python.framework import common_shapes
    from tensorflow.python.framework import ops

    path = os.path.dirname(__file__)
    rime_lib_path = os.path.join(path, 'rime_ops', 'rime.so')
    rime_lib = tf.load_op_library(rime_lib_path)

    # Register shape operators
    # TODO(sperkins) Find some other more sensible place to do this
    ops.RegisterShape("Phase")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("BSqrt")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("EBeam")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("EKBSqrt")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("GaussShape")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("SersicShape")(common_shapes.call_cpp_shape_fn)
    ops.RegisterShape("SumCoherencies")(common_shapes.call_cpp_shape_fn)

    return rime_lib

