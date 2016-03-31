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

from base_solver import BaseSolver

class CUDAArrayDescriptor(object):
    """ Descriptor class for pycuda.gpuarrays on the GPU """
    def __init__(self, record_key, default=None):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class ContextWrapper(object):
    """ Context Manager Wrapper for CUDA Contexts! """
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        """ Pushed the wrapped context onto the stack """
        self.context.push()
        return self

    def __exit__(self,type,value,traceback):
        """ Pop when we're done """
        import pycuda.driver as cuda
        cuda.Context.pop()

class CUDASolver(BaseSolver):
    """ Solves the RIME using CUDA """
    def __init__(self, *args, **kwargs):
        import pycuda.driver as cuda
        
        super(CUDASolver, self).__init__(*args, **kwargs)

        ctx = kwargs.get('context', None)

        if ctx is None:
            raise ValueError("Expected a CUDA 'context' keyword.")

        # Create a context wrapper
        self.context = ContextWrapper(ctx)

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        A = super(CUDASolver, self).register_array(
            name, shape, dtype, registrant, **kwargs)

        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        # Create descriptors on the class instance, even though members
        # may not necessarily be created on object instances. This is so
        # that if someone registers an array but doesn't ask for it to be
        # created, we have control over it, if at some later point they wish
        # to do a
        #
        # slvr.blah = ...
        #

        # TODO, there's probably a better way of figuring out if a descriptor
        # is set on the class
        #if not hasattr(CUDASolver, A.name):
        if A.name not in CUDASolver.__dict__:
            setattr(CUDASolver, A.name, CUDAArrayDescriptor(record_key=A.name))

        # We don't use gpuarray.zeros, since it fails for
        # a zero-length array. This is kind of bad since
        # the gpuarray returned by gpuarray.empty() doesn't
        # have GPU memory allocated to it.
        with self.context as ctx:
            """
            # Query free memory on this context
            (free_mem,total_mem) = cuda.mem_get_info()

            montblanc.log.debug("Allocating GPU memory "
                "of size {s} for array '{n}'. {f} free "
                "{t} total on device.".format(n=name,
                    s=self.fmt_bytes(self.array_bytes(A)),
                    f=self.fmt_bytes(free_mem),
                    t=self.fmt_bytes(total_mem)))

            """

            gpu_ary = gpuarray.empty(shape=A.shape, dtype=A.dtype)
            
            setattr(self, A.name, gpu_ary)

        return A
