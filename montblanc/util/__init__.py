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

import montblanc

from ary_dim_eval import eval_expr, eval_expr_names_and_nrs
from sky_model_parser import parse_sky_model

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

def dict_array_bytes(ary, props):
    """
    Return the number of bytes required by an array

    Arguments
    ---------------
    ary : dict
        Dictionary representation of an array
    props : dict
        A dictionary of key-values, used to replace any
        string values in the array with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    the array.
    """
    shape = shape_from_str_tuple(ary['shape'], props)
    dtype = dtype_from_str(ary['dtype'], props)

    return array_bytes(shape, dtype)

def dict_array_bytes_required(arrays, props):
    """
    Return the number of bytes required by
    a dictionary of arrays.

    Arguments
    ---------------
    arrays : list
        A list of dictionaries defining the arrays
    props : dict
        A dictionary of key-values, used to replace any
        string values in the arrays with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    all the arrays.
    """
    return np.sum([dict_array_bytes(ary, props)
        for ary in arrays])

def viable_dim_config(bytes_available, arrays, props,
        dim_ord, nsolvers=1):
    """
    Returns the number of timesteps possible, given the registered arrays
    and a memory budget defined by bytes_available

    Arguments
    ----------------
    bytes_available : int
        The memory budget, or available number of bytes
        for solving the problem.
    arrays : list
        List of dictionaries describing the arrays
    props : dict
        Dictionary containing key-values that will be used
        to replace any string representations of dimensions
        and types. slvr.get_properties() will return something
        suitable.
    dim_ord : list
        list of dimension string names that the problem should be
        subdivided by. e.g. ['ntime', 'nbl', 'nchan'].
        Multple dimensions can be reduced simultaneously using
        the following syntax 'nbl&na'. This is mostly useful for
        the baseline-antenna equivalence.

    Keyword Arguments
    ----------------------------
    nsolvers : int
        Number of solvers to budget for. Defaults to one.

    Returns
    ----------
    A tuple (boolean, dict). The boolean is True if the problem
    can fit within the supplied budget, False otherwise.
    THe dictionary contains the reduced dimensions as key and
    the reduced size as value.
    e.g. (True, { 'time' : 1, 'nbl' : 1 })

    For a dim_ord = ['ntime', 'nbl', 'nchan'], this method will try and fit
    a ntime x nbl x nchan problem into the available number of bytes.
    If this is not possible, it will first set ntime=1, and then try fit an
    1 x nbl x nchan problem into the budget, then a 1 x 1 x nchan
    problem.
    """

    if not isinstance(dim_ord, list):
        raise TypeError('dim_ord should be a list')

    # Don't accept non-negative memory budgets
    if bytes_available < 0:
        bytes_available = 0

    modified_dims = {}

    bytes_used = dict_array_bytes_required(arrays, props)

    # While more bytes are used than are available, set
    # dimensions to one in the order specified by the
    # dim_ord argument.
    while bytes_used > bytes_available:
        try:
            dims = dim_ord.pop(0).strip().split('&')
        except IndexError:
            # No more dimensions available for reducing
            # the problem size. Unable to fit the problem
            # within the specified memory budget
            return False, modified_dims

        # Can't fit everything into memory,
        # Set dimensions to 1 and re-evaluate
        for dim in dims:
            modified_dims[dim] = 1
            props[dim] = 1

        bytes_used = dict_array_bytes_required(arrays, props)

    return True, modified_dims


def viable_timesteps(bytes_available, arrays, props):
    """
    Returns the number of timesteps possible, given the registered arrays
    and a memory budget defined by bytes_available
    """

    # Don't accept non-negative memory budgets
    if bytes_available < 0:
        bytes_available = 0

    # Figure out which arrays have an ntime dimension
    has_time = np.array([ \
        t.sshape.count('ntime') > 0 for t in arrays.values()])

    # Get the shape product of each array, EXCLUDING any ntime dimension,
    # multiplied by the size of the array type in bytes.
    products = np.array([array_bytes(
        shape_from_str_tuple(t.sshape, props,
            ignore=['ntime']),
        t.dtype)
        for t in arrays.values()])

    # TODO: Remove duplicate code paths
    # This really replicates solver.bytes_required
    bytes_required = np.array([array_bytes(
        shape_from_str_tuple(t.sshape, props),
        t.dtype)
        for t in arrays.values()]).sum()

    # Determine a linear expression for the bytes
    # required which varies by timestep. y = a + b*x
    a = np.sum(np.logical_not(has_time)*products)
    b = np.sum(has_time*products)

    # Check that if we substitute ntime for x, we agree on the
    # memory requirements
    assert a + b*props['ntime'] == bytes_required

    # Given the number of bytes available,
    # how many timesteps can we fit in our budget?
    return (bytes_available - a + b - 1) // b

def dtype_from_str(sdtype, props):
    """
    Substitutes string dtype parameters
    using a a property dictionary

    Parameters
    ----------
        sdtype :
            string defining the dtype
        props

    Returns
        sdtype if it isn't a string
        props[sdtype] otherwise

    """

    if not isinstance(sdtype, str):
        return sdtype

    return props[sdtype]

def shape_from_str_tuple(sshape, variables, ignore=None):
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

    >>> print self.shape_from_str_tuple((4,'na','ntime'),ignore=['ntime'])
    (4, 3)
    """
    if ignore is None: ignore = []

    if not isinstance(sshape, tuple) and not isinstance(sshape, list):
        raise TypeError, 'sshape argument must be a tuple or list'

    if not isinstance(ignore, list):
        raise TypeError, 'ignore argument must be a list'

    return tuple([int(eval_expr(v,variables)) if isinstance(v,str) else int(v)
        for v in sshape if v not in ignore])

def shape_list(l,shape,dtype):
    """ Shape a list of lists into the appropriate shape and data type """
    return np.array(l, dtype=dtype).reshape(shape)

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
    n_one = shape_from_str_tuple(s_one, variables)
    n_two = [eval_expr(d,variables)
        if isinstance(d,str) else d for d in sshape_two]

    def f(ary): return np.reshape(ary, n_one).transpose(t_idx).reshape(n_two)

    return f

from montblanc.src_types import SOURCE_VAR_TYPES, POINT_TYPE

def default_sources(**kwargs):
    """
    Returns a dictionary mapping source types
    to number of sources. If the number of sources
    for the source type is supplied in the kwargs
    these will be placed in the dictionary.

    e.g. if we have 'point', 'gaussian' and 'sersic'
    source types, then

    default_sources(point=10, gaussian=20)

    will return a dict {'point': 10, 'gaussian': 20, 'sersic': 0}
    """
    S = {}
    total = 0

    invalid_types = [t for t in kwargs.keys() if t not in SOURCE_VAR_TYPES]

    for t in invalid_types:
        montblanc.log.warning('Source type %s is not yet '
            'implemented in montblanc. '
            'Valid source types are %s' % (t, SOURCE_VAR_TYPES.keys()))

    # Zero all source types
    for k, v in SOURCE_VAR_TYPES.iteritems():
        # Try get the number of sources for this source
        # from the kwargs
        value = kwargs.get(k, 0)

        try:
            value = int(value)
        except ValueError:
            raise TypeError(('Supplied value %s '
                'for source %s cannot be '
                'converted to an integer') % \
                    (value, k))    

        total += value
        S[k] = value

    # Add a point source if no others exist
    if total == 0:
        S[POINT_TYPE] = 1

    return S

def sources_to_nr_vars(sources):
    """
    Converts a source type to number of sources mapping into
    a source numbering variable to number of sources mapping.

    If, for example, we have 'point', 'gaussian' and 'sersic'
    source types, then passing the following dict as an argument
    
    sources_to_nr_vars({'point':10, 'gaussian': 20})
    
    will return a new dict

    {'npsrc': 10, 'ngsrc': 20, 'nssrc': 0 }
    """

    sources = default_sources(**sources)

    try:
        return { SOURCE_VAR_TYPES[name]: nr for name, nr in sources.iteritems() }
    except KeyError as e:
        raise KeyError((
            'No source type ''%s'' is '
            'registered. Valid source types '
            'are %s') % (e, SOURCE_VAR_TYPES.keys()))

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
