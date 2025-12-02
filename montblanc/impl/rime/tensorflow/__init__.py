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
    def __pkg_resource_import(): # needed for Python3.6 compatibility
        from pkg_resources import working_set
        from pkg_resources import Requirement

        import tensorflow as tf
        import os

        path = pjoin('ext', 'rime.so')
        mbloc = pjoin(working_set.find(Requirement.parse('montblanc')).location, "montblanc")
        rime_lib_path = pjoin(mbloc, path)
        if not os.path.isfile(rime_lib_path):
            from montblanc import ext
            rime_lib_path = os.path.join(os.path.dirname(ext.__file__), 'rime.so')
        if not os.path.isfile(rime_lib_path):
            raise RuntimeError(f"Montblanc backend not found: '{rime_lib_path}'. Have you compiled the backend?")
        return tf.load_op_library(rime_lib_path)
    def __implib_resource_import():
        import tensorflow as tf
        import os
        import importlib
        rime_lib_path = os.path.join(os.path.dirname(importlib.import_module('montblanc').__file__),
                                     'ext', 'rime.so')
        if not os.path.isfile(rime_lib_path):
            from montblanc import ext
            rime_lib_path = os.path.join(os.path.dirname(ext.__file__), 'rime.so')
        if not os.path.isfile(rime_lib_path):
            raise RuntimeError(f"Montblanc backend not found: '{rime_lib_path}'. Have you compiled the backend?")
        return tf.load_op_library(rime_lib_path)
    try:
        return __implib_resource_import()
    except (ImportError, ModuleNotFoundError): # fallback for Python3.6
        return __pkg_resource_import()

