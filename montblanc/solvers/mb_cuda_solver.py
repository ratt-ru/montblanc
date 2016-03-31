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
import types

from cuda_solver import CUDASolver
from rime_solver import RIMESolver
from montblanc.config import SolverConfig as Options

class MontblancCUDASolver(RIMESolver, CUDASolver):
    """ Solves the RIME using CUDA """
    def __init__(self, slvr_cfg):
        super(MontblancCUDASolver, self).__init__(slvr_cfg=slvr_cfg,
            context=slvr_cfg.get(Options.CONTEXT, None))
        self.pipeline = slvr_cfg.get('pipeline')

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        A = super(MontblancCUDASolver, self).register_array(
            name, shape, dtype, registrant, **kwargs)

        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        # Create an empty array
        cpu_ary = np.empty(shape=A.shape, dtype=A.dtype)                
        data_source = self._slvr_cfg[Options.DATA_SOURCE]

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
                kwargs.get(Options.DATA_SOURCE_DEFAULT, None))               

        # We don't use gpuarray.zeros, since it fails for
        # a zero-length array. This is kind of bad since
        # the gpuarray returned by gpuarray.empty() doesn't
        # have GPU memory allocated to it.
        with self.context:
            # If the array length is non-zero initialise it
            if (data_source != Options.DATA_SOURCE_EMPTY and
                np.product(A.shape) > 0):

                getattr(self, name).set(cpu_ary)

        # Should we create a setter for this property?
        transfer_method = kwargs.get('transfer_method', True)

        # OK, we got a boolean for the kwarg, create a default transfer method
        if isinstance(transfer_method, types.BooleanType) and transfer_method is True:
            # Create the transfer method
            def transfer(self, npary):
                self.check_array(A.name, npary)
                with self.context:
                    getattr(self,A.name).set(npary)

            transfer_method = types.MethodType(transfer,self)
        # Otherwise, we can just use the supplied kwarg
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
        """ % (A.name,A.name)

        # Should we create a getter for this property?
        retrieve_method = kwargs.get('retrieve_method', True)

        # OK, we got a boolean for the kwarg, create a default retrieve method
        if isinstance(retrieve_method, types.BooleanType) and retrieve_method is True:
            # Create the retrieve method
            def retrieve(self):
                with self.context:
                    return getattr(self,A.name).get()

            retrieve_method = types.MethodType(retrieve,self)
        # Otherwise, we can just use the supplied kwarg
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
        """ % (A.name,A.name)


        return A

    def solve(self):
        """ Solve the RIME """
        with self.context as ctx:
            self.pipeline.execute(self)

    def initialise(self):
        with self.context as ctx:
            self.pipeline.initialise(self)

    def shutdown(self):
        """ Stop the RIME solver """
        with self.context as ctx:
            self.pipeline.shutdown(self)             

    def transfer_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'transfer_' + name

    def retrieve_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'retrieve_' + name            