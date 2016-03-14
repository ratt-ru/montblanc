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

from cffi import FFI

from montblanc.src_types import (
    source_types,
    source_nr_vars,
    default_sources,
    sources_to_nr_vars)

# Name of the structure type
_STRUCT_TYPE = 'rime_const_data'
# Name of the structure type
_STRUCT_PTR_TYPE = _STRUCT_TYPE + ' *'

class RimeConstDefinition(object):
    """
    Holds the cffi object defining the C structure
    of the RIME constant data which will be passed
    to GPUs.
    """
    def __init__(self, slvr):
        self._ffi = FFI()
        # Parse the structure
        self._ffi.cdef(self._struct(slvr))
        self._cstr = self._struct(slvr)

    @staticmethod
    def _struct(slvr):
        """
        Returns a string containing
        the C definition of the
        RIME constant data structure
        """
        def _emit_struct_field_str(type, name):
            return ' '*4 + type + ' ' + name + ';'

        l = ['typedef struct {']
        l.extend([_emit_struct_field_str('unsigned int', v)
            for v in slvr.dimensions().iterkeys()])
        l.append('} ' + _STRUCT_TYPE + ';')
        return '\n'.join(l)

    def struct_size(self):
        """
        Returns the size in bytes of
        the RIME constant data structure
        """
        return self._ffi.sizeof(_STRUCT_TYPE)

    def wrap(self, ndary):
        assert ndary.nbytes == self.struct_size(), \
            ('The size of the supplied array {as} does '
            'not match that of the constant data structure {ds}.') \
                .format(ws=ndary.nbytes, ds=self.struct_size())

        # Create a cdata object by wrapping ndary
        # and cast to the structure type
        return self._ffi.cast(_STRUCT_PTR_TYPE, self._ffi.from_buffer(ndary))

    def __str__(self):
        return self._cstr

class RimeConstStruct(object):
    def __init__(self, rime_const_def, ndary):
        self._def = rime_const_def
        self._ndary = ndary
        self._cdata = self._def.wrap(ndary)

    def ndary(self):
        return self._ndary

    def update(self, slvr, sum_nsrc=False):
        """
        Update the RIME constant data structure
        """
        # Iterate through our dimension data, setting
        # any relevant attributes on cdata
        for name, dim in slvr.dimensions().iteritems():
            setattr(self._cdata, name, dim.extents[1])

        if sum_nsrc is True and hasattr(self._cdata, 'nsrc'):
            # Set cdata.nsrc by summing each source type
            setattr(self._cdata, 'nsrc',
                sum([getattr(self._cdata, s)
                    for s in source_nr_vars()]))

    def string_def(self):
        return self._def.__str__()

    def cdata(self):
        """ Return the cdata object """
        return self._cdata

def create_rime_const_data(slvr, context):
    """ Creates RIME constant data object """
    import pycuda.driver as cuda

    # Create a structure definition
    struct_def = RimeConstDefinition(slvr)

    # Allocate some page-locked memory within the
    # given CUDA context
    with context:
        ndary = cuda.pagelocked_empty(shape=struct_def.struct_size(),
            dtype=np.int8)

    # Construct the constant structure object from the
    # definition and the ndary. Update it with the
    # current solver values
    const_data = RimeConstStruct(struct_def, ndary)
    const_data.update(slvr)

    return const_data



