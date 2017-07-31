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
import re

import montblanc

from parallactic_angles import parallactic_angles

from montblanc.src_types import (
    source_types,
    source_nr_vars,
    default_sources,
    sources_to_nr_vars,
    source_range,
    source_range_tuple,
    source_range_slices)

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

def blocks_required(N, threads_per_block):
    """
    Returns the number of blocks required, given
    N, the total number of threads, and threads_per_block
    """
    return (N + threads_per_block - 1) / threads_per_block

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

def random_float(shape, dtype):
    return np.random.random(size=shape).astype(dtype)

def random_complex(shape, ctype):
    if ctype == np.complex64:
        dtype = np.float32
    elif ctype == np.complex128:
        dtype = np.float64
    else:
        raise TypeError("Invalid complex type {ct}".format(ct=ctype))

    return random_float(shape, dtype) + 1j*random_float(shape, dtype)

def random_like(ary=None, shape=None, dtype=None):
    """
    Returns a random array of the same shape and type as the
    supplied array argument, or the supplied shape and dtype
    """
    if ary is not None:
        shape, dtype = ary.shape, ary.dtype
    elif shape is None or dtype is None:
        raise ValueError((
            'random_like(ary, shape, dtype) must be supplied '
            'with either an array argument, or the shape and dtype '
            'of the desired random array.'))

    if np.issubdtype(dtype, np.complexfloating):
        return (np.random.random(size=shape) + \
            np.random.random(size=shape)*1j).astype(dtype)
    else:
        return np.random.random(size=shape).astype(dtype)

def flatten(nested):
    """ Return a flatten version of the nested argument """
    flat_return = list()

    def __inner_flat(nested,flat):
        for i in nested:
            __inner_flat(i, flat) if isinstance(i, list) else flat.append(i)
        return flat

    __inner_flat(nested,flat_return)

    return flat_return

def dict_array_bytes(ary, template):
    """
    Return the number of bytes required by an array

    Arguments
    ---------------
    ary : dict
        Dictionary representation of an array
    template : dict
        A dictionary of key-values, used to replace any
        string values in the array with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    the array.
    """
    shape = shape_from_str_tuple(ary['shape'], template)
    dtype = dtype_from_str(ary['dtype'], template)

    return array_bytes(shape, dtype)

def dict_array_bytes_required(arrays, template):
    """
    Return the number of bytes required by
    a dictionary of arrays.

    Arguments
    ---------------
    arrays : list
        A list of dictionaries defining the arrays
    template : dict
        A dictionary of key-values, used to replace any
        string values in the arrays with concrete integral
        values

    Returns
    -----------
    The number of bytes required to represent
    all the arrays.
    """
    return np.sum([dict_array_bytes(ary, template)
        for ary in arrays])

__DIM_REDUCTION_RE = re.compile(    # Capture Groups and Subgroups
    "^\s*(?P<name>[A-Za-z0-9_]*?)"  # 1.   Dimension name
    "(?:\s*?=\s*?"                  # 2.   White spaces and =
        "(?P<value>[0-9]*?)"        # 2.1  A value
        "(?P<percent>\%?)"          # 2.2  Possibly followed by a percentage
    ")?\s*?$")                      #      Capture group 2 possibly occurs

def viable_dim_config(bytes_available, arrays, template,
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
    template : dict
        Dictionary containing key-values that will be used
        to replace any string representations of dimensions
        and types. slvr.template_dict() will return something
        suitable.
    dim_ord : list
        list of dimension string names that the problem should be
        subdivided by. e.g. ['ntime', 'nbl', 'nchan'].
        Multple dimensions can be reduced simultaneously using
        the following syntax 'nbl&na'. This is mostly useful for
        the baseline-antenna equivalence.
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

    One can specify reductions for specific dimensions.
    For e.g. ['ntime=20', 'nbl=1&na=2', 'nchan=50%']

    will reduce ntime to 20, but no lower. nbl=1&na=2 sets
    both nbl and na to 1 and 2 in the same operation respectively.
    nchan=50\% will continuously halve the nchan dimension
    until it reaches a value of 1.
    """

    if not isinstance(dim_ord, list):
        raise TypeError('dim_ord should be a list')

    # Don't accept non-negative memory budgets
    if bytes_available < 0:
        bytes_available = 0

    modified_dims = {}
    T = template.copy()

    bytes_used = dict_array_bytes_required(arrays, T)*nsolvers

    # While more bytes are used than are available, set
    # dimensions to one in the order specified by the
    # dim_ord argument.
    while bytes_used > bytes_available:
        try:
            dims = dim_ord.pop(0)
            montblanc.log.debug('Applying reduction {s}. '
                'Bytes available: {a} used: {u}'.format(
                    s=dims,
                    a=fmt_bytes(bytes_available),
                    u=fmt_bytes(bytes_used)))
            dims = dims.strip().split('&')
        except IndexError:
            # No more dimensions available for reducing
            # the problem size. Unable to fit the problem
            # within the specified memory budget
            return False, modified_dims

        # Can't fit everything into memory,
        # Lower dimensions and re-evaluate
        for dim in dims:
            match = re.match(__DIM_REDUCTION_RE, dim)

            if not match:
                raise ValueError(
                    "{d} is an invalid dimension reduction string "
                    "Valid strings are for e.g. "
                    "'ntime', 'ntime=20' or 'ntime=20%'"
                        .format(d=dim))

            dim_name = match.group('name')
            dim_value = match.group('value')
            dim_percent = match.group('percent')
            dim_value = 1 if dim_value is None else int(dim_value)

            # Attempt reduction by a percentage
            if dim_percent == '%':
                dim_value = int(T[dim_name] * int(dim_value) / 100.0)
                if dim_value < 1:
                    # This can't be reduced any further
                    dim_value = 1
                else:
                    # Allows another attempt at reduction
                    # by percentage on this dimension
                    dim_ord.insert(0, dim)

            # Apply the dimension reduction
            if T[dim_name] > dim_value:
                modified_dims[dim_name] = dim_value
                T[dim_name] = dim_value
            else:
                montblanc.log.info(('Ignored reduction of {d} '
                    'of size {s} to {v}. ').format(
                        d=dim_name, s=T[dim_name], v=dim_value))

        bytes_used = dict_array_bytes_required(arrays, T)*nsolvers

    return True, modified_dims

def dtype_from_str(sdtype, template):
    """
    Substitutes string dtype parameters
    using a a property dictionary

    Parameters
    ----------
        sdtype :
            string defining the dtype. e.g. 'ft'
        template:
            Dictionary contain actual types, {'ft' : np.float32' }

    Returns
        sdtype if it isn't a string
        template[sdtype] otherwise

    """

    if not isinstance(sdtype, str):
        return sdtype

    return template[sdtype]

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

def redistribute_threads(blockdimx, blockdimy, blockdimz,
    dimx, dimy, dimz):
    """
    Redistribute threads from the Z dimension towards the X dimension.
    Also clamp number of threads to the problem dimension size,
    if necessary
    """

    # Shift threads from the z dimension
    # into the y dimension
    while blockdimz > dimz:
        tmp = blockdimz // 2
        if tmp < dimz:
            break

        blockdimy *= 2
        blockdimz = tmp

    # Shift threads from the y dimension
    # into the x dimension
    while blockdimy > dimy:
        tmp = blockdimy // 2
        if tmp < dimy:
            break

        blockdimx *= 2
        blockdimy = tmp

    # Clamp the block dimensions
    # if necessary
    if dimx < blockdimx:
        blockdimx = dimx

    if dimy < blockdimy:
        blockdimy = dimy

    if dimz < blockdimz:
        blockdimz = dimz

    return blockdimx, blockdimy, blockdimz


def register_default_dimensions(cube, slvr_cfg):
    """ Register the default dimensions for a RIME solver """

    import montblanc.src_types as mbs

    # Pull out the configuration options for the basics
    autocor = slvr_cfg['auto_correlations']

    ntime = 10
    na = 7
    nbands = 1
    nchan = 16
    npol = 4

    # Infer number of baselines from number of antenna,
    nbl = nr_of_baselines(na, autocor)

    if not npol == 4:
        raise ValueError("npol set to {}, but only 4 polarisations "
                         "are currently supported.")

    # Register these dimensions on this solver.
    cube.register_dimension('ntime', ntime,
        description="Timesteps")
    cube.register_dimension('na', na,
        description="Antenna")
    cube.register_dimension('nbands', nbands,
        description="Bands")
    cube.register_dimension('nchan', nchan,
        description="Channels")
    cube.register_dimension('npol', npol,
        description="Polarisations")
    cube.register_dimension('nbl', nbl,
        description="Baselines")

    # Register dependent dimensions
    cube.register_dimension('npolchan', nchan*npol,
        description='Polarised channels')
    cube.register_dimension('nvis', ntime*nbl*nchan,
        description='Visibilities')

    # Convert the source types, and their numbers
    # to their number variables and numbers
    # { 'point':10 } => { 'npsrc':10 }
    src_cfg = default_sources()
    src_nr_vars = sources_to_nr_vars(src_cfg)
    # Sum to get the total number of sources
    cube.register_dimension('nsrc', sum(src_nr_vars.itervalues()),
        description="Sources (Total)")

    # Register the individual source types
    for nr_var, nr_of_src in src_nr_vars.iteritems():
        cube.register_dimension(nr_var, nr_of_src,
            description='{} sources'.format(mbs.SOURCE_DIM_TYPES[nr_var]))

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
        self.context.pop()