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
from weakref import WeakKeyDictionary

from rime_solver import RIMESolver
import montblanc.util as mbu
from montblanc.config import SolverConfig as Options

class NumpyArrayDescriptor(object):
    """ Descriptor class for NumPy ndarrays arrays on the CPU """
    def __init__(self, record_key, default=None):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        instance.check_array(self.record_key, value)
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class NumpySolver(RIMESolver):
    def __init__(self, slvr_cfg):
        super(NumpySolver, self).__init__(slvr_cfg)    

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        """
        Register an array with this Solver object.

        Arguments
        ----------
            name : string
                name of the array.
            shape : integer/string or tuple of integers/strings
                Shape of the array.
            dtype : data-type
                The data-type for the array.
            registrant : string
                Name of the entity registering this array.

        Keyword Arguments
        -----------------
            page_locked : boolean
                True if the 'name_cpu' ndarray should be allocated as
                a page-locked array.
            aligned : boolean
                True if the 'name_cpu' ndarray should be allocated as
                an page-aligned array.

        Returns
        -------
            A dictionary describing this array.
        """

        A = super(NumpySolver, self).register_array(
            name, shape, dtype, registrant, **kwargs)

       # Attribute names
        cpu_name = mbu.cpu_name(A.name)

        # Create descriptors on the class instance, even though members
        # may not necessarily be created on object instances. This is so
        # that if someone registers an array but doesn't ask for it to be
        # created, we have control over it, if at some later point they wish
        # to do a
        #
        # slvr.blah_cpu = ...
        #

        # TODO, there's probably a better way of figuring out if a descriptor
        # is set on the class
        #if not hasattr(NumpySolver, cpu_name):
        if cpu_name not in NumpySolver.__dict__:
            setattr(NumpySolver, cpu_name,
                NumpyArrayDescriptor(record_key=A.name))

        page_locked = kwargs.get('page_locked', False)
        aligned = kwargs.get('aligned', False)

        if page_locked or aligned:
            raise ValueError("Page-locked and aligned arrays "
                "are not currently handled!")

        # Create an empty array
        cpu_ary = np.empty(shape=A.shape, dtype=A.dtype)                
        data_source =self._slvr_cfg[Options.DATA_SOURCE]

        # If we're creating test data, initialise the array with
        # data from the test key, don't initialise if we've been
        # explicitly told the array should be empty, otherwise
        # set the defaults
        if data_source == Options.DATA_SOURCE_TEST:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_TEST, None))
        elif data_source == Options.DATA_SOURCE_EMPTY:
            pass
        else:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_DEFAULTS, None))

        # Create the attribute on the solver
        setattr(self, cpu_name, cpu_ary)

        return A