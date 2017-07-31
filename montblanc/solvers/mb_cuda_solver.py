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

import types

import numpy as np

import hypercube as hc
from hypercube.array_factory import (
    create_local_arrays_on_cube,
    create_local_arrays,
    generic_stitch,
    gpuarray_factory)

from rime_solver import RIMESolver
import montblanc.util as mbu

class MontblancCUDASolver(RIMESolver):
    """ Solves the RIME using CUDA """
    def __init__(self, slvr_cfg):
        super(MontblancCUDASolver, self).__init__(slvr_cfg=slvr_cfg)

        self.pipeline = slvr_cfg.get('pipeline')
        self.context = mbu.ContextWrapper(slvr_cfg.get(Options.CONTEXT))

    def initialise_gpu_array(self, name, array, data_source):
        gpu_ary = getattr(self, name)
        cpu_ary = np.empty(shape=gpu_ary.shape, dtype=gpu_ary.dtype)

        if data_source == Options.DATA_SOURCE_TEST:
            self.init_array(name, cpu_ary,
                array.get(Options.DATA_SOURCE_TEST, None))
        elif data_source == Options.DATA_SOURCE_EMPTY:
            pass
        else:
            self.init_array(name, cpu_ary,
                array.get(Options.DATA_SOURCE_DEFAULT, None))

        # We don't use gpuarray.zeros, since it fails for
        # a zero-length array. This is kind of bad since
        # the gpuarray returned by gpuarray.empty() doesn't
        # have GPU memory allocated to it.
        # If the array length is non-zero initialise it
        if (data_source != Options.DATA_SOURCE_EMPTY and
            np.product(gpu_ary.shape) > 0):

            gpu_ary.set(cpu_ary)

        # Should we create a setter for this property?
        transfer_method = array.get('transfer_method', True)

        # OK, we got a boolean create a default transfer method
        if (isinstance(transfer_method, types.BooleanType) and
            transfer_method is True):

            # Create the transfer method
            def transfer(self, npary):
                with self.context:
                    gpu_ary.set(npary)

            transfer_method = types.MethodType(transfer,self)
        # Otherwise, we can just use the supplied method
        elif isinstance(transfer_method, types.MethodType):
            pass
        else:
            raise TypeError(('transfer_method keyword argument set '
                'to an invalid type %s') % (type(transfer_method)))

        # Name the transfer method
        transfer_method_name = self.transfer_method_name(name)
        setattr(self,  transfer_method_name, transfer_method)
        # Create a docstring!
        getattr(transfer_method, '__func__').__doc__ = \
        """
        Transfers the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (name,name)

        # Should we create a getter for this property?
        retrieve_method = array.get('retrieve_method', True)

        # OK, we got a boolean create a default retrieve method
        if (isinstance(retrieve_method, types.BooleanType)
            and retrieve_method is True):

            # Create the retrieve method
            def retrieve(self):
                with self.context:
                    return gpu_ary.get()

            retrieve_method = types.MethodType(retrieve,self)
        # Otherwise, we can just use the supplied method
        elif isinstance(retrieve_method, types.MethodType):
            pass
        else:
            raise TypeError(('retrieve_method keyword argument set '
                'to an invalid type %s') % (type(retrieve_method)))

        # Name the retrieve method
        retrieve_method_name = self.retrieve_method_name(name)
        setattr(self,  retrieve_method_name, retrieve_method)
        # Create a docstring!
        getattr(retrieve_method, '__func__').__doc__ = \
        """
        Retrieve the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (name,name)

    def create_arrays(self, ignore=None, supplied=None):
        """
        Create any necessary arrays on the solver.

        Arguments
        ---------
            ignore : list
                List of array names to ignore.
            supplied : dictionary
                A dictionary of supplied arrays to create
                on the solver, keyed by name. Note that
                these arrays will not be initialised by
                montblanc, it is the responsibility of the
                user to initialise them.
        """

        import pycuda.driver as cuda

        if ignore is None:
            ignore = []

        if supplied is None:
            supplied = {}

        reified_arrays = self.arrays(reify=True)
        create_arrays = self._arrays_to_create(reified_arrays,
            ignore=ignore, supplied=supplied)

        with self.context:
            # Create local arrays on the cube
            create_local_arrays_on_cube(self, create_arrays,
                array_stitch=generic_stitch,
                array_factory=gpuarray_factory)

            self._validate_supplied_arrays(reified_arrays, supplied)

            # Stitch the supplied arrays onto the cube
            generic_stitch(self, supplied)

            # Get our data source
            data_source = self._slvr_cfg[Options.DATA_SOURCE]

            for name, array in create_arrays.iteritems():
                self.initialise_gpu_array(name, array, data_source)

    def solve(self):
        """ Solve the RIME """
        with self.context:
            self.pipeline.execute(self)

    def initialise(self):
        with self.context:
            self.pipeline.initialise(self)

    def shutdown(self):
        """ Stop the RIME solver """
        with self.context:
            self.pipeline.shutdown(self)

    def transfer_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'transfer_' + name

    def retrieve_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'retrieve_' + name