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
import os
import re
import string
import sys
import types

import numpy as np
from astropy.io import fits

from hypercube import HyperCube

import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow.sources.source_provider import SourceProvider

class FitsFilenameTemplate(string.Template):
    """
    Overrides the ${identifer} braced pattern in the string Template
    with a $(identifier) braced pattern expected by FITS beam filename
    schema
    """
    pattern = r"""
    %(delim)s(?:
      (?P<escaped>%(delim)s)   |   # Escape sequence of two delimiters
      (?P<named>%(id)s)        |   # delimiter and a Python identifier
      \((?P<braced>%(id)s)\)   |   # delimiter and a braced identifier
      (?P<invalid>)                # Other ill-formed delimiter exprs
    )
    """ % { 'delim' : re.escape(string.Template.delimiter),
        'id' : string.Template.idpattern }

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

def _create_filenames(filename_schema):
    """
    Returns a dictionary of beam filename pairs,
    keyed on correlation,from the cartesian product
    of correlations and real, imaginary pairs

    Given 'beam_$(corr)_$(reim).fits' returns:
    {
      'xx' : ('beam_xx_re.fits', 'beam_xx_im.fits'),
      'xy' : ('beam_xy_re.fits', 'beam_xy_im.fits'),
      ...
      'yy' : ('beam_yy_re.fits', 'beam_yy_im.fits'),
    }

    Given 'beam_$(CORR)_$(REIM).fits' returns:
    {
      'xx' : ('beam_XX_RE.fits', 'beam_XX_IM.fits'),
      'xy' : ('beam_XY_RE.fits', 'beam_XY_IM.fits'),
      ...
      'yy' : ('beam_YY_RE.fits', 'beam_YY_IM.fits'),
    }

    """
    template = FitsFilenameTemplate(filename_schema)

    def _re_im_filenames(corr, template):
        return tuple(template.substitute(
            corr=corr.lower(), CORR=corr.upper(),
            reim=ri.lower(), REIM=ri.upper())
                for ri in REIM)

    return collections.OrderedDict(
        (c, _re_im_filenames(c, template))
        for c in CORRELATIONS)

def _open_fits_files(filenames):
    kw = { 'mode' : 'update', 'memmap' : False }

    def _fh(fn):
        """ Returns a filehandle or None if file does not exist """
        return fits.open(fn, **kw) if os.path.exists(fn)  else None

    return collections.OrderedDict(
            (corr, tuple(_fh(fn) for fn in files))
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

def _create_axes(filenames, file_dict):
    """ Create a FitsAxes object """

    try:
        # Loop through the file_dictionary, finding the
        # first open FITS file.
        f = iter(f for tup in file_dict.itervalues()
            for f in tup if f is not None).next()
    except StopIteration as e:
        raise (ValueError("No FITS files were found. "
            "Searched filenames: '{f}'." .format(
                f=filenames.values())),
                    None, sys.exc_info()[2])


    # Create a FitsAxes object
    axes = FitsAxes(f[0].header)

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
    """
    Feeds holography cubes from a series of eight FITS files matching a
    filename_schema. A schema of 'beam_$(corr)_$(reim).fits' produces:

    ['beam_xx_re.fits', 'beam_xx_im.fits',
     'beam_xy_re.fits', 'beam_xy_im.fits',
      ...
      'beam_yy_re.fits', 'beam_yy_im.fits']

    while 'beam_$(CORR)_$(REIM).fits'

    ['beam_XX_RE.fits', 'beam_XX_IM.fits',
      'beam_XY_RE.fits', 'beam_XY_IM.fits',
      ...
      'beam_YY_RE.fits', 'beam_YY_IM.fits'\


    Missing files will result in zero values for that correlation
    and real/imaginary component. The shape of the FITS data will be
    inferrred from the first file found and subsequent files should match
    that shape.
    """
    def __init__(self, filename_schema):
        """
        """
        self._filename_schema = filename_schema
        self._filenames = _create_filenames(filename_schema)
        self._files = _open_fits_files(self._filenames)
        self._axes = _create_axes(self._filenames, self._files)
        self._dim_indices = (l_ax, m_ax, f_ax) = _cube_dim_indices(
            self._axes)
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

        self._dim_updates_indicated = False

    def name(self):
        return self._name

    @cache_fits_read
    def ebeam(self, context):
        """ Feeds the ebeam cube """
        if context.shape != self.shape:
            raise ValueError("Partial feeding of the "
                "beam cube is not yet supported.")

        ebeam = np.empty(context.shape, context.dtype)

        # Iterate through the correlations,
        # assigning real and imaginary data, if present,
        # otherwise zeroing the correlation
        for i, (re, im) in enumerate(self._files.itervalues()):
            ebeam[:,:,:,i].real[:] = re[0].data.T if re is not None else 0
            ebeam[:,:,:,i].imag[:] = im[0].data.T if im is not None else 0

        return ebeam

    @cache_fits_read
    def beam_extents(self, context):
        """ Return the beam extents """
        return self._cube_extents.flatten()

    @cache_fits_read
    def beam_freq_map(self, context):
        """ Return the frequency map associated with the beam """
        lower_freq, upper_freq = self._cube_extents[:,2]
        return np.linspace(lower_freq, upper_freq, context.shape[0])

    def updated_dimensions(self):
        # Dimension updates bave been indicated, don't send them again
        if self._dim_updates_indicated is True:
            return ()

        self._dim_updates_indicated = True
        return [self._cube.dimension(k,copy=False)
            for k in ('beam_lw', 'beam_mh', 'beam_nud')]

    @property
    def filename_schema(self):
        return self._filename_schema

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

    def __str__(self):
        return self.__class__.__name__