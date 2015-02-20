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
import math

from ary_dim_eval import eval_expr, eval_expr_names_and_nrs

def nr_of_baselines(na, auto_correlations=False):
    """
    Compute the number of baselines for the
    given number of antenna. Can specify whether
    auto-correlations should be taken into
    account
    """
    m = (na-1) if auto_correlations is False else (na+1)
    return (na*m)//2

def nr_of_antenna(nbl, auto_correlations=False):
    """
    Compute the number of antenna for the
    given number of baselines. Can specify whether
    auto-correlations should be taken into
    account
    """
    t = 1 if auto_correlations is False else -1
    return int(t + math.sqrt(1 + 8*nbl)) // 2

def cpu_name(name):
    """ Constructs a name for the CPU version of the array """
    return name + '_cpu'

def gpu_name(name):
    """ Constructs a name for the GPU version of the array """
    return name + '_gpu'

def transfer_method_name(name):
    """ Constructs a transfer method name, given the array name """
    return 'transfer_' + name

def shape_name(name):
    """ Constructs a name for the array shape member, based on the array name """
    return name + '_shape'

def dtype_name(name):
    """ Constructs a name for the array data-type member, based on the array name """
    return name + '_dtype'

def setter_name(name):
    """ Constructs a name for the property, based on the property name """
    return 'set_' + name

def fmt_array_line(name,size,dtype,cpu,gpu,shape):
    """ Format array parameters on an 80 character width line """
    return '%-*s%-*s%-*s%-*s%-*s%-*s' % (
        20,name,
        10,size,
        15,dtype,
        4,cpu,
        4,gpu,
        20,shape)

def fmt_property_line(name,dtype,value,default):
    return '%-*s%-*s%-*s%-*s' % (
        20,name,
        10,dtype,
        20,value,
        20,default)

def fmt_bytes(nbytes):
    """ Returns a human readable string, given the number of bytes """
    for x in ['B','KB','MB','GB']:
        if nbytes < 1024.0:
            return "%3.1f%s" % (nbytes, x)
        nbytes /= 1024.0

    return "%.1f%s" % (nbytes, 'TB')

def array_bytes(shape, dtype):
    """ Estimates the memory in bytes required for an array of the supplied shape and dtype """
    return np.product(shape)*np.dtype(dtype).itemsize

def rethrow_attribute_exception(e):
    """
    Rethrows an attribute exception with more informative text.
    Used in CPU code for cases when the solver doesn't have
    the desired arrays configured.
    """
    raise AttributeError('%s. The appropriate numpy array has not '
        'been set on the solver object. You need to set '
        'store_cpu=True on your solver object '
        'as well as call the transfer_* method for this to work.' % e)

def flatten(nested):
    """ Return a flatten version of the nested argument """
    flat_return = list()

    def __inner_flat(nested,flat):
        for i in nested:
            __inner_flat(i, flat) if isinstance(i, list) else flat.append(i)
        return flat

    __inner_flat(nested,flat_return)

    return flat_return

def get_numeric_shape(sshape, variables, ignore=None):
    """
    Substitutes string values in the supplied shape parameter
    with integer variables stored in a dictionary

    Parameters
    ----------
    sshape : tuple/string composed of integers and strings.
        The strings should related to integral properties
        registered with this Solver object
    variables : dictionary
        Keys with associated integer values. Used to replace
        string values within the tuple
    ignore : list
        A list of tuple strings to ignore

    >>> print self.get_numeric_shape((4,'na','ntime'),ignore=['ntime'])
    (4, 3)
    """
    if ignore is None: ignore = []

    if not isinstance(sshape, tuple) and not isinstance(sshape, list):
        raise TypeError, 'sshape argument must be a tuple or list'

    if not isinstance(ignore, list):
        raise TypeError, 'ignore argument must be a list'

    return tuple([int(eval_expr(v,variables)) if isinstance(v,str) else int(v)
        for v in sshape if v not in ignore])

def array_convert_function(sshape_one, sshape_two, variables):
    """ Return a function defining the conversion process between two NumPy
    arrays of different shapes """
    if not isinstance(sshape_one, tuple): sshape_one = (sshape_one,)
    if not isinstance(sshape_two, tuple): sshape_two = (sshape_two,)

    s_one = flatten([eval_expr_names_and_nrs(d) if isinstance(d,str) else d
        for d in sshape_one])
    s_two = flatten([eval_expr_names_and_nrs(d) if isinstance(d,str) else d
        for d in sshape_two])

    if len(s_one) != len(s_two):
        raise ValueError, ('Flattened shapes %s and %s '\
            'do not have the same length. '
            'Original shapes were %s and %s') % \
            (s_one, s_two, sshape_one, sshape_two)

    # Reason about the transpose
    t_idx = tuple([s_one.index(v) for v in s_two])

    # Figure out the actual numeric shape values to use
    n_one = get_numeric_shape(s_one, variables)
    n_two = [eval_expr(d,variables)
        if isinstance(d,str) else d for d in sshape_two]

    def f(ary): return np.reshape(ary, n_one).transpose(t_idx).reshape(n_two)

    return f

import pycuda.driver as cuda

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
        cuda.Context.pop()
