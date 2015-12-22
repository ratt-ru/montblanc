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

import traceback
import types

from distarray.globalapi.distarray import DistArray
import numpy as np

import montblanc

# Standard NumPy arrays
CPU_ARY_NUMPY = 'numpy'
CPU_ARY_NUMPY_DESCRIPTION = '{p}Standard NumPy array created with numpy.empty.'

# NumPy arrays with a page-locked base
CPU_ARY_PAGELOCKED = 'page_locked'
CPU_ARY_PAGELOCKED_DESCRIPTION = (
    '{p}NumPy array wrapper around pinned/page-locked CPU memory. \n'
    '{p}Allocated with pycuda.driver.pagelocked_empty.')

# NumPy arrays with an aligned base
CPU_ARY_ALIGNED = 'aligned'
CPU_ARY_ALIGNED_DESCRIPTION = (
    '{p}NumPy array wrapper around aligned CPU memory. \n'
    '{p}Allocated with pycuda.driver.aligned_empty.')

# distarray distributed arrays
CPU_ARY_DISTARRAY = 'distarray'
CPU_ARY_DISTARRAY_DESCRIPTION = (
    '{p}NumPy-like distributed array '
    'from the distarray package.')

CPU_ARY_MAP = {
    CPU_ARY_NUMPY : CPU_ARY_NUMPY_DESCRIPTION,
    CPU_ARY_PAGELOCKED : CPU_ARY_PAGELOCKED_DESCRIPTION,
    CPU_ARY_ALIGNED : CPU_ARY_ALIGNED_DESCRIPTION,
    CPU_ARY_DISTARRAY : CPU_ARY_DISTARRAY_DESCRIPTION
}

CPU_ARY_TYPES = [CPU_ARY_NUMPY, CPU_ARY_PAGELOCKED, CPU_ARY_ALIGNED,
    CPU_ARY_DISTARRAY]

# Create a formatted description of the CPU_ARY_MAP
ARRAY_TYPE_DESCRIPTION ='\n'.join(["{p}'{n}'\n{d}".format(
        p=' '*4, n=name, d=desc.format(p=' '*8))
    for name, desc in CPU_ARY_MAP.iteritems()])

WARNING_MESSAGE = ("A default array in key '{key}'' "
    "was copied into {method} array '{name}', "
    "instead of directly assigned. "
    "This may be inefficient in terms "
    "of time and memory, "
    "but is otherwise harmless.")

INIT_KWARG = 'init'
_INIT_KWARG_TYPE = 'string, ndarray, function'
_INIT_KWARG_DESCRIPTION = '{p}Default array value'

DISTARRAY_CONSTRUCTOR_KWARG = 'distarray_constructor'
_DISTARRAY_CONSTRUCTOR_KWARG_TYPE = 'lambda or function'
_DISTARRAY_CONSTRUCTOR_KWARG_DESCRIPTION = (
    "{p}A lambda or function that takes zero arguments\n"
    "{p} and returns a distarray object.")


CONTEXT_KWARG = 'context'
_CONTEXT_KWARG_TYPE = 'Montblanc ContextWrapper for PyCUDA'
_CONTEXT_KWARG_DESCRIPTION = (
    "{p}ContextWrapper wrapping a PyCUDA context.\n"
    "{p}Required for '{pl}' and '{al}' arrays.").format(
        p='{p}', pl=CPU_ARY_PAGELOCKED, al=CPU_ARY_ALIGNED)

_KWARG_MAP = {
    INIT_KWARG : (
        _INIT_KWARG_TYPE,
        _INIT_KWARG_DESCRIPTION),
    CONTEXT_KWARG : (
        _CONTEXT_KWARG_TYPE,
        _CONTEXT_KWARG_DESCRIPTION),
    DISTARRAY_CONSTRUCTOR_KWARG : (
        _DISTARRAY_CONSTRUCTOR_KWARG_TYPE,
        _DISTARRAY_CONSTRUCTOR_KWARG_DESCRIPTION),
}

# Create a formatted description of the _KWARG_MAP
_KWARG_DESCRIPTION ='\n'.join(["{p}{n} : {t}\n{d}".format(
        p=' '*4, n=name, t=stype, d=desc.format(p=' '*8))
    for name, (stype, desc) in _KWARG_MAP.iteritems()])

def init_array(ary, value):
    """ Initialise ary using value """

    # If value is some sort of function, execute it
    # and extract the return value
    if isinstance(value, (types.LambdaType, types.FunctionType, types.MethodType)):
        value = value(ary)

        # If we get None at this point, bail out, as we
        # assume that the function will have modified
        # ary in place. This avoids the None case lower down
        if value is None:
            return

    # Returned value is the same as the original, don't assign
    if value is ary:
        return

    # No defaults were supplied
    elif value is None:
        ary.fill(0)

    # Got an ndarray, try set it equal
    elif isinstance(ary, np.ndarray) and isinstance(value, np.ndarray):
        try:
            ary[:] = value
        except BaseException as e:
            raise ValueError(("Tried to assign array with "
                "value NumPy array, but this failed "
                "with {e}").format(e=traceback.format_exc()))
    elif isinstance(ary, DistArray) and isinstance(value, np.ndarray):
        # Just assign if array shapes match
        if ary.shape == value.shape:
            try:
                ary[:] = value
            except BaseException as e:
                raise ValueError(("Tried to assign array with "
                    "value NumPy array, but this failed "
                    "with {e}").format(e=traceback.format_exc()))
        # Assume an array broadcast. As distarray does not yet
        # handle this, try the broadcast on the remote's local array
        elif np.product(ary.shape) > np.product(value.shape):
            def assign_on_remote_local(local_ary, value):
                local_ary.ndarray[:] = value

            ary.context.apply(assign_on_remote_local,
                args=(ary.key, value))
        # distarray doesn't handle this case, just ignore it.
        elif np.product(ary.shape) == 0:
            pass
        # Otherwise we don't know how to handle this
        else:
            raise ValueError(("Unable to assign array of "
                "shape {ns} and type {nt} to a distarray "
                "with shape {ds} and {dt},").format(
                    ns=value.shape, nt=value.dtype,
                    ds=ary.shape, dt=ary.dtype))

    # Assume some sort of value has been supplied
    # Give it to NumPy
    else:
        try:
            ary.fill(value)
        except BaseException as e:
            raise ValueError(("Tried to fill array with "
                "value '{v}', but NumPy\'s fill function "
                "failed with {e}").format(v=value, e=traceback.format_exc()))

def cpu_array_factory(shape, dtype, name=None,
    ary_type=None, **kwargs):
    """
    Create an empty CPU array of specified shape and dtype.

    Arguments
    ---------
    shape : tuple
        Tuple of integers describing the shape of the array dimensions.
    dtype : NumPy data type
        Describes the data type associated with this array.
    ary_type : string
        The type of array to be created.

    {ary_type_desc}

    Keyword Arguments
    -----------------
    {kwargs_desc}

    Returns
    -------
    An array of the specified type.

    """

    # Handle default arguments
    if ary_type is None:
        ary_type = CPU_ARY_NUMPY

    # Convert any True boolean types to NumPy arrays
    if isinstance(ary_type, types.BooleanType) and ary_type is True:
        ary_type = CPU_ARY_NUMPY

    # Get any default values for the array
    init = kwargs.get(INIT_KWARG, None)

    def get_context(mem_type, kwargs):
        ctx = kwargs.get(CONTEXT_KWARG, None)

        if not ctx:
            raise ValueError(('A CUDA context is required '
                'for allocating {m} memory, '
                'but none was supplied.').format(m=mem_type))

        return ctx


    # Standard NumPy arrays
    if ary_type == CPU_ARY_NUMPY:
        # If the type, shape and dtype of the init value matches ...
        if isinstance(init, np.ndarray) and \
            init.shape == shape and init.dtype == dtype:

            return init

        A = np.empty(shape=shape, dtype=dtype)
        init_array(A, init)
        return A

    # Page-locked NumPy array 
    elif ary_type == CPU_ARY_PAGELOCKED:
        import pycuda.driver as cuda

        with get_context(ary_type, kwargs):
            A = cuda.pagelocked_empty(shape=shape, dtype=dtype)
            init_array(A, init)
            return A

    # Memory aligned NumPy array
    elif ary_type == CPU_ARY_ALIGNED:
        import pycuda.driver as cuda

        with get_context(ary_type, kwargs):
            A = cuda.aligned_empty(shape=shape, dtype=dtype)
            init_array(A, init)
            return A

    # Distributed distarray
    elif ary_type == CPU_ARY_DISTARRAY:
        da_lambda = kwargs.get(DISTARRAY_CONSTRUCTOR_KWARG)
        A = da_lambda
        init_array(A, init)
        return A

    raise ValueError(("Array type '{at}' is not supported. "
        "Valid array types are {vat}").format(
        at=ary_type, vat=CPU_ARY_MAP.keys()))

import textwrap

# Format the docstring
cpu_array_factory.__doc__ = textwrap.dedent(
        cpu_array_factory.__doc__).format(
            ary_type_desc=ARRAY_TYPE_DESCRIPTION,
            kwargs_desc=_KWARG_DESCRIPTION)
