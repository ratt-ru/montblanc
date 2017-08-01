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

        # Check for custom irregular grid format.
        # Currently only implemented for FREQ dimension.
        irregular_grid = [[header.get('G%s%d' % (self._ctype[i], j), None)
            for j in range(1, self._naxis[i]+1)]
            for i in range(ndims)]

        # Irregular grids are only valid if values exist for all grid points
        valid = [all(x is not None for x in irregular_grid[i])
            for i in range(ndims)]

        def _regular_grid(a, i):
            """ Construct a regular grid from a FitsAxes object and index """
            R = np.arange(0.0, float(a.naxis[i]))
            return (R - a.crpix[i])*a.cdelta[i] + a.crval[i]

        # Set up the grid
        self._grid = [_regular_grid(self, i) if not valid[i]
            else np.asarray(irregular_grid[i]) for i in range(ndims)]

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
    def grid(self):
        return self._grid

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

CIRCULAR_CORRELATIONS = ('rr', 'rl', 'lr', 'll')
LINEAR_CORRELATIONS = ('xx', 'xy', 'yx', 'yy')
REIM = ('re', 'im')

def _create_filenames(filename_schema, feed_type):
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

    if feed_type == 'linear':
        CORRELATIONS = LINEAR_CORRELATIONS
    elif feed_type == 'circular':
        CORRELATIONS = CIRCULAR_CORRELATIONS
    else:
        raise ValueError("Invalid feed_type '{}'. "
            "Should be 'linear' or 'circular'")

    return collections.OrderedDict(
        (c, _re_im_filenames(c, template))
        for c in CORRELATIONS)

def _open_fits_files(filenames):
    """
    Given a {correlation: filename} mapping for filenames
    returns a {correlation: file handle} mapping
    """
    kw = { 'mode' : 'update', 'memmap' : False }

    def _fh(fn):
        """ Returns a filehandle or None if file does not exist """
        return fits.open(fn, **kw) if os.path.exists(fn) else None

    return collections.OrderedDict(
            (corr, tuple(_fh(fn) for fn in files))
        for corr, files in filenames.iteritems() )

def _cube_extents(axes, l_ax, m_ax, f_ax, l_sign, m_sign):
    # List of (lower, upper) extent tuples for the given dimensions
    it = zip((l_ax, m_ax, f_ax), (l_sign, m_sign, 1.0))
    # Get the extents, flipping the sign on either end if required
    extent_list = [tuple(s*e for e in axes.extents[i]) for i, s in it]

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

def _axis_and_sign(ax_str):
    """ Extract axis and sign from given axis string """
    return (ax_str[1:], -1.0) if ax_str[0] == '-' else (ax_str, 1.0)

class FitsBeamSourceProvider(SourceProvider):
    """
    Feeds holography cubes from a series of eight FITS files matching a
    filename_schema. A schema of :code:`'beam_$(corr)_$(reim).fits'`
    matches:

    .. code-block:: python

        ['beam_xx_re.fits', 'beam_xx_im.fits',
         'beam_xy_re.fits', 'beam_xy_im.fits',
          ...
          'beam_yy_re.fits', 'beam_yy_im.fits']

    while :code:`'beam_$(CORR)_$(REIM).fits'` matches

    .. code-block:: python

        ['beam_XX_RE.fits', 'beam_XX_IM.fits',
          'beam_XY_RE.fits', 'beam_XY_IM.fits',
          ...
          'beam_YY_RE.fits', 'beam_YY_IM.fits']


    Missing files will result in zero values for that correlation
    and real/imaginary component. The shape of the FITS data will be
    inferred from the first file found and subsequent files should match
    that shape.

    The type of correlation will be derived from the feed type.
    Currently, linear :code:`['xx', 'xy', 'yx', 'yy']` and
    circular :code:`['rr', 'rl', 'lr', 'll']` are supported.
    """
    def __init__(self, filename_schema, l_axis=None, m_axis=None):
        """
        Constructs a FitsBeamSourceProvider object

        Parameters
        ----------
            filename_schema : str
                See :py:class:`.FitsBeamSourceProvider` for valid schemas
            l_axis : str
                FITS axis interpreted as the L axis. `L` and `X` are
                sensible values here. `-L` will invert the coordinate
                system on that axis.
            m_axis : str
                FITS axis interpreted as the M axis. `M` and `Y` are
                sensible values here. `-M` will invert the coordinate
                system on that axis.
        """
        l_axis, l_sign = _axis_and_sign('L' if l_axis is None else l_axis)
        m_axis, m_sign = _axis_and_sign('M' if m_axis is None else m_axis)

        self._l_axis = l_axis
        self._l_sign = l_sign
        self._m_axis = m_axis
        self._m_sign = m_sign

        self._fits_dims = fits_dims = (l_axis, m_axis, 'FREQ')
        self._beam_dims = ('beam_lw', 'beam_mh', 'beam_nud')

        self._filename_schema = filename_schema
        self._name = "FITS Beams '{s}'".format(s=filename_schema)

        # Have we initialised this object?
        self._initialised = False

    def _initialise(self, feed_type="linear"):
        """
        Initialise the object by generating appropriate filenames,
        opening associated file handles and inspecting the FITS axes
        of these files.
        """
        self._filenames = filenames = _create_filenames(self._filename_schema,
                                                        feed_type)
        self._files = files = _open_fits_files(filenames)
        self._axes = axes = _create_axes(filenames, files)
        self._dim_indices = dim_indices = l_ax, m_ax, f_ax = tuple(
            axes.iaxis(d) for d in self._fits_dims)

        # Complain if we can't find required axes
        for i, ax in zip(dim_indices, self._fits_dims):
            if i == -1:
                raise ValueError("'%s' axis not found!" % ax)

        self._cube_extents = _cube_extents(axes, l_ax, m_ax, f_ax,
            self._l_sign, self._m_sign)
        self._shape = tuple(axes.naxis[d] for d in dim_indices) + (4,)
        self._beam_freq_map = axes.grid[f_ax]

        # Now describe our dimension sizes
        self._dim_updates = [(n, axes.naxis[i]) for n, i
            in zip(self._beam_dims, dim_indices)]

        self._initialised = True

    def name(self):
        """ Name of this Source Provider """
        return self._name

    def init(self, init_context):
        """ Perform any initialisation """
        self._initialise(init_context.cfg['polarisation_type'])

    def ebeam(self, context):
        """ ebeam cube data source """
        if context.shape != self.shape:
            raise ValueError("Partial feeding of the "
                "beam cube is not yet supported %s %s." % (context.shape, self.shape))

        ebeam = np.empty(context.shape, context.dtype)

        # Iterate through the correlations,
        # assigning real and imaginary data, if present,
        # otherwise zeroing the correlation
        for i, (re, im) in enumerate(self._files.itervalues()):
            ebeam[:,:,:,i].real[:] = 0 if re is None else re[0].data.T
            ebeam[:,:,:,i].imag[:] = 0 if im is None else im[0].data.T

        return ebeam

    def beam_extents(self, context):
        """ Beam extent data source """
        return self._cube_extents.flatten().astype(context.dtype)

    def beam_freq_map(self, context):
        """ Beam frequency map data source """
        return self._beam_freq_map.astype(context.dtype)

    def updated_dimensions(self):
        """ Indicate dimension sizes """
        return self._dim_updates

    @property
    def filename_schema(self):
        """ Filename schema """
        return self._filename_schema

    @property
    def shape(self):
        """ Shape of the beam cube """
        return self._shape

    def close(self):
        if not hasattr(self, "_files"):
            return

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