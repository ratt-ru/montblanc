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

__rime_lib = None

def load_tf_lib():
    """ Load the tensorflow library """

    import montblanc
    import tensorflow as tf

    global __rime_lib

    if __rime_lib is not None:
        return __rime_lib

    mb_path = montblanc.get_montblanc_path()
    rime_lib_path = os.path.join(mb_path, 'extensions', 'tensorflow', 'rime.so')
    __rime_lib = tf.load_op_library(rime_lib_path)

    return __rime_lib

