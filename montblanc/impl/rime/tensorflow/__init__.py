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

def load_tf_lib(rime_lib_path=None):
    """ Load the tensorflow library """
    import pkg_resources

    import tensorflow as tf

    if rime_lib_path is None:
        from os.path import join as pjoin
        rime_lib_path = pjoin('ext', 'rime.so')
        rime_lib_path = pkg_resources.resource_filename("montblanc",
                                                        rime_lib_path)

    return tf.load_op_library(rime_lib_path)

