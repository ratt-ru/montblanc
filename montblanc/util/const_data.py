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

from montblanc.enums import (
    DIMDATA)

from montblanc.src_types import (
    source_types,
    source_nr_vars,
    default_sources,
    sources_to_nr_vars)

_SPACE = ' '*4
# Name of the field type
_FIELD_TYPE = 'dim_field'
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
        self._cstr = self._struct(slvr)
        self._ffi.cdef(self._struct(slvr))

    @staticmethod
    def _struct(slvr):
        """
        Returns a string containing
        the C definition of the
        RIME constant data structure
        """
        def _emit_struct_field_str(name):
            return _SPACE + '{t} {n};'.format(t=_FIELD_TYPE, n=name)

        # Define our field structure. Looks something like
        # typedef struct {
        #     unsigned int global_size;
        #     unsigned int local_size;
        #     unsigned int extents[2];
        # } _FIELD_TYPE;
        l = ['typedef struct  {']
        l.extend([_SPACE + 'unsigned int {n};'.format(n=n)
            for n in (DIMDATA.GLOBAL_SIZE,
                DIMDATA.LOCAL_SIZE,
                DIMDATA.EXTENTS+'[2]')])
        l.append('}} {t};'.format(t=_FIELD_TYPE))

        # Define our constant data structure. Looks something like
        # typedef struct {
        #     _FIELD_TYPE ntime;
        #     _FIELD_TYPE na;
        #     ....
        # } _STRUCT_TYPE;

        l.append('typedef struct {')
        l.extend([_emit_struct_field_str(n) for n
            in slvr.dimensions().iterkeys()])
        l.append('} ' + _STRUCT_TYPE + ';')

        # Join with newlines and return the string
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
        """ Returns the wrapped numpy array backing the constant data struct """
        return self._ndary

    def update(self, slvr, sum_nsrc=False):
        """
        Update the RIME constant data structure
        """
        # Iterate through our dimension data, setting
        # any relevant attributes on cdata
        for name, dim in slvr.dimensions().iteritems():
            cdim = getattr(self._cdata, name)

            setattr(cdim, DIMDATA.LOCAL_SIZE,
                getattr(dim, DIMDATA.LOCAL_SIZE))

            setattr(cdim, DIMDATA.GLOBAL_SIZE,
                getattr(dim, DIMDATA.GLOBAL_SIZE))

            setattr(cdim, DIMDATA.EXTENTS, 
                getattr(dim, DIMDATA.EXTENTS))

        from montblanc.slvr_config import (
            SolverConfigurationOptions as Options)

        # If 'nsrc' exists set it by by summing each source type
        if sum_nsrc is True:
            cdim = getattr(self._cdata, Options.NSRC, None)

            if cdim:
                # This performs an element-wise sum over each sources extents
                S = map(sum, zip(*[getattr(getattr(self._cdata, s),
                        DIMDATA.EXTENTS)
                    for s in source_nr_vars()]))

                setattr(cdim, DIMDATA.EXTENTS, S)

    def string_def(self):
        """ Return the C string definition of the structure """
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



