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

def load_tf_lib():
    """ Load the tensorflow library """
    from os.path import join as pjoin
    from pkg_resources import working_set
    from pkg_resources import Requirement

    import tensorflow as tf

    path = pjoin('ext', 'rime.so')
    mbloc = pjoin(working_set.find(Requirement.parse('montblanc')).location, "montblanc")
    rime_lib_path = pjoin(mbloc, path)
    return tf.load_op_library(rime_lib_path)

