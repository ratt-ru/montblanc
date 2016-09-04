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

import collections
import functools
import sys
import types

import numpy as np
from astropy.io import fits

from hypercube import HyperCube

import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow.sources.source_provider import SourceProvider

class FitsAxes(object):
    """
    FitsAxes object, inspired by Tigger's FITSAxes
    """
    def __init__(self, header):
        self._ndims = ndims = header['NAXIS']

        # Extract header information for each dimension
        axr = range(1, ndims+1)
        self._naxis = [header.get('NAXIS%d'%n)      for n in axr]
        self._ctype = [header.get('CTYPE%d'%n, n)   for n in axr]
        self._crval = [header.get('CRVAL%d'%n, 0)   for n in axr]
        self._crpix = [header.get('CRPIX%d'%n)-1    for n in axr]
        self._cdelta = [header.get('CDELT%d'%n, 1)  for n in axr]
        self._cunit = [header.get('CUNIT%d'%n, '').strip().upper()
            for n in axr]

        # Copy original CRVAL and CRDELTA in case they are scaled
        self._scale = [1.0 for n in axr]
        self._crval0 = [v for v in self._crval]
        self._cdelta0 = [v for v in self._cdelta]

        # Map axis names to integers
        self._iaxis = {n: i for i, n in enumerate(self._ctype)}

    @property
    def ndims(self):
        return self._ndims

    def iaxis(self, name):
        try:
            return self._iaxis[name]
        except KeyError:
            return -1

    @property
    def crpix(self):
        return self._crpix

    @property
    def naxis(self):
        return self._naxis

    @property
    def crval(self):
        return self._crval

    @property
    def cdelta(self):
        return self._cdelta

    @property
    def crval0(self):
        return self._crval0

    @property
    def cdelta0(self):
        return self._cdelta0

    @property
    def cunit(self):
        return self._cunit

    @property
    def ctype(self):
        return self._ctype

    @property
    def scale(self):
        return self._scale

    @property
    def extents(self):
        f = lambda v, i: (v - self.crpix[i])*self.cdelta[i] + self.crval[i]
        return [tuple(f(v, i)  for v in (0, self.naxis[i]-1) )
            for i in range(self.ndims)]

    def set_axis_scale(self, index, scale):
        self.scale[index] = scale
        self.crval[index] = self.crval0[index]*scale
        self.cdelta[index] = self.cdelta0[index]*scale

CORRELATIONS = ('xx', 'xy', 'yx', 'yy')
REIM = ('re', 'im')
BEAM_DIMS = ('L', 'M', 'FREQ')

def _create_filenames(base_filename):
    """
    Infer a dictionary of beam filename pairs,
    keyed on correlation,from the cartesian product
    of correlations and real, imaginary pairs

    Given 'beam' as the base_filename, returns something like
    {
      'xx' : ('beam_xx_re.fits', 'beam_xx_im.fits'),
      'xy' : ('beam_xy_re.fits', 'beam_xy_im.fits'),
      ...
      'yy' : ('beam_yy_re.fits', 'beam_yy_im.fits'),
    }
    """
    def _re_im_filenames(corr, base):
        return tuple('{b}_{c}_{ri}.fits'.format(
                b=base, c=corr, ri=ri)
            for ri in REIM)

    return collections.OrderedDict(
        (c, _re_im_filenames(c, base_filename))
        for c in CORRELATIONS)

def _filename_schema(base_filename):
    """
    Given the base_filename, print out a string
    illustrating the naming scheme for the FITS filenames
    """
    return '{b}_{{{c}}}_{{{ri}}}.fits'.format(
        b=base_filename,
        c='/'.join(CORRELATIONS),
        ri='/'.join(REIM))

def _open_fits_files(filenames):
    open_kwargs = { 'mode' : 'update', 'memmap' : False }

    return collections.OrderedDict(
            (corr, tuple(fits.open(fn, **open_kwargs) for fn in files))
        for corr, files in filenames.iteritems() )

def _cube_dim_indices(axes):
    # Identify L, M and FREQ axis indices
    l_ax, m_ax, f_ax = (axes.iaxis(d) for d in BEAM_DIMS)
    # Try use X and Y if we an't find L and M
    l_ax, m_ax = (axes.iaxis(d) if i == -1 else i
        for i, d in zip([l_ax, m_ax], ['X', 'Y']))

    return (l_ax, m_ax, f_ax)

def _cube_extents(axes, l_ax, m_ax, f_ax):
    # List of (lower, upper) extent tuples for the given dimensions
    extent_list = [axes.extents[i] for i in (l_ax, m_ax, f_ax)]

    # Return [[l_low, u_low, f_low], [l_high, u_high, f_high]]
    return np.array(extent_list).T

def _create_axes(file_dict):
    """ Create a FitsAxes object """
    re0, im0 = file_dict.values()[0]

    # Create a Cattery FITSAxes object
    axes = FitsAxes(re0[0].header)

    # Scale any axes in degrees to radians
    for i, u in enumerate(axes.cunit):
        if u == 'DEG':
            axes.cunit[i] = 'RAD'
            axes.set_axis_scale(i, np.pi/180.0)

    return axes

def cache_fits_read(method):
    """
    Decorator for caching FitsBeamDataSource source function return values

    Create a key index for the proxied array in the SourceContext.
    Iterate over the array shape descriptor e.g.
    (beam_lw, beam_mh, beam_nud, 4)
    returning tuples containing the lower and upper extents
    of string dimensions. Takes (0, d) in the case of an integer
    dimensions.
    """

    @functools.wraps(method)
    def memoizer(self, context):
        D = context.dimensions(copy=False)
        # (lower, upper) else (0, d)
        idx = ((D[d].lower_extent, D[d].upper_extent) if d in D
            else (0, d) for d in context.array(context.name).shape)
        # Construct the key for the above index
        key = tuple(i for t in idx for i in t)
        # Access the sub-cache for this array
        array_cache = self._cache[context.name]

        # Cache miss, call the function
        if key not in array_cache:
            array_cache[key] = method(self, context)

        return array_cache[key]

    return memoizer


class FitsBeamSourceProvider(SourceProvider):
    """ Feeds holography cubes from FITS files """
    def __init__(self, base_beam_filename):
        self._base_beam_filename = base_beam_filename
        self._filenames = _create_filenames(base_beam_filename)
        self._files = _open_fits_files(self._filenames)
        self._axes = _create_axes(self._files)
        self._dim_indices = (l_ax, m_ax, f_ax) = _cube_dim_indices(self._axes)
        self._name = "FITS Beams '{s}'".format(s=self.filename_schema)

        # Complain if we can't find required axes
        for i, ax in enumerate(zip(self._dim_indices, BEAM_DIMS)):
            if i == -1:
                raise ValueError("'%s' axis not found!" % ax)

        self._cube_extents = _cube_extents(self._axes, l_ax, m_ax, f_ax)
        self._shape = tuple(self._axes.naxis[d] for d in self._dim_indices) + (4,)

        self._cache = collections.defaultdict(dict)

        # Now create a hypercube describing the dimensions
        self._cube = cube = HyperCube()

        cube.register_dimension('beam_lw',
            self._axes.naxis[l_ax],
            description='E Beam cube l width')

        cube.register_dimension('beam_mh',
            self._axes.naxis[m_ax],
            description='E Beam cube m height')

        cube.register_dimension('beam_nud',
            self._axes.naxis[f_ax],
            description='E Beam cube frequency depth')

    def name(self):
        return self._name

    @cache_fits_read
    def ebeam(self, context):
        """ Feeds the ebeam cube """
        if context.shape != self.shape:
            raise ValueError("Partial feeding of the "
                "beam cube is not yet supported.")

        ebeam = np.empty(context.shape, context.dtype)

        for i, (re, im) in enumerate(self._files.itervalues()):
            ebeam[:,:,:,i].real[:] = re[0].data.T
            ebeam[:,:,:,i].imag[:] = im[0].data.T

        return ebeam

    def beam_extents(self, context):
        """ Return the beam extents """
        return self._cube_extents.flatten()

    def updated_dimensions(self):
        return [self._cube.dimension(k,copy=False)
            for k in ('beam_lw', 'beam_mh', 'beam_nud')]

    @property
    def filename_schema(self):
        return _filename_schema(self._base_beam_filename)

    @property
    def base_beam_filename(self):
        return self._base_beam_filename

    @property
    def shape(self):
        """ Shape of the beam cube """
        return self._shape

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        self.clear_cache()

        for re, im in self._files.itervalues():
            re.close()
            im.close()

        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()