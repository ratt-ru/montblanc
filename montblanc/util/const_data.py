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

from cffi import FFI

from montblanc.src_types import (
    source_types,
    source_nr_vars,
    default_sources,
    sources_to_nr_vars)

# Foreign function interface object
_ffi = FFI()
# Name of the structure type
_STRUCT_TYPE = 'rime_const_data'
# Name of the structure type
_STRUCT_PTR_TYPE = _STRUCT_TYPE + ' *'
# Variables contained within the structure
_STRUCT_DATA_VAR_LIST = ['ntime', 'nbl', 'na', 'nchan', 'npolchan', 'nsrc']

def rime_const_data_members():
    """ Returns a list of struct data members """
    return _STRUCT_DATA_VAR_LIST + source_nr_vars()

def _emit_struct_field_str(type, name):
    return ' '*4 + type + ' ' + name + ';'

def rime_const_data_struct():
    """
    Returns a string containing
    the C definition of the
    RIME constant data structure
    """
    l = ['typedef struct {']
    l.extend([_emit_struct_field_str('unsigned int', v)
        for v in rime_const_data_members()])
    l.append('} ' + _STRUCT_TYPE + ';')
    return '\n'.join(l)

# Parse the structure
_ffi.cdef(rime_const_data_struct())

def rime_const_data_size():
    """
    Returns the size in bytes of
    the RIME constant data structure
    """
    return _ffi.sizeof(_STRUCT_TYPE)

def wrap_rime_const_data(ndary):
    """
    Returns a cffi cdata structure object that
    uses the supplied ndary as it's storage space

    ndary.nbytes should be equal to rime_const_data_size()
    """
    struct_size = rime_const_data_size()

    assert ndary.nbytes == struct_size, \
        ('The size of the supplied array {as} does '
        'not match that of the constant data structure {ds}.') \
            .format(ws=ndary.nbytes, ds=struct_size)

    # Create a cdata object by wrapping ndary
    # and cast to the structure type
    return _ffi.cast(_STRUCT_PTR_TYPE, _ffi.from_buffer(ndary))

def update_rime_const_data(slvr, rime_const_data, sum_nsrc=False):
    """
    Update the RIME constant data structure
    """
    # Iterate through our dimension data, setting
    # any relevant attributes on rime_const_data
    for name, dim in slvr.dims.iteritems():
        if hasattr(rime_const_data, name):
            setattr(rime_const_data, name, dim.extents[1])

    if sum_nsrc is True and hasattr(rime_const_data, 'nsrc'):
        # Set rime_const_data.nsrc by summing each source type
        setattr(rime_const_data, 'nsrc',
            sum([getattr(rime_const_data, s)
                for s in source_nr_vars()]))
