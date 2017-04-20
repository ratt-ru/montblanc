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
import types

import numpy as np

import montblanc.util as mbu
from montblanc.config import RimeSolverConfig as Options
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources.source_provider import SourceProvider

def cache_ms_read(method):
    """
    Decorator for caching MSRimeDataSource source function return values

    Create a key index for the proxied array in the SourceContext.
    Iterate over the array shape descriptor e.g. (ntime, nbl, 3)
    returning tuples containing the lower and upper extents
    of string dimensions. Takes (0, d) in the case of an integer
    dimensions.
    """

    @functools.wraps(method)
    def memoizer(self, context):
        # Defer to the method if no caching is enabled
        if not self._is_cached:
            return method(self, context)

        # Construct the key for the given index
        idx = context.array_extents(context.name)
        key = tuple(i for t in idx for i in t)
        # Access the sub-cache for this array
        array_cache = self._cache[context.name]

        # Cache miss, call the function
        if key not in array_cache:
            array_cache[key] = method(self, context)

        return array_cache[key]

    return memoizer

class MSSourceProvider(SourceProvider):
    """
    Source Provider that retrieves input data from a
    MeasurementSet
    """

    def __init__(self, manager, vis_column=None, cache=False):
        """
        Constructs an MSSourceProvider object

        Parameters
        ----------
        manager: :py:class:`.MeasurementSetManager`
            The :py:class:`.MeasurementSetManager` used to access
            the Measurement Set.
        vis_column: str
            Column from which observed visibilities will be read
        cache: bool
            Indicates whether data will be cached on this
            Source Provider.
        """
        self._manager = manager
        self._name = "Measurement Set '{ms}'".format(ms=manager.msname)

        self._vis_column = 'DATA' if vis_column is None else vis_column

        self._is_cached = cache

        self._cache = collections.defaultdict(dict)

    def name(self):
        return self._name

    def updated_dimensions(self):
        # Defer to manager's method
        return self._manager.updated_dimensions()

    @cache_ms_read
    def phase_centre(self, context):
        # [0][0] because (a) we select only 1 row
        #                (b) assumes a NUM_POLY of 1
        data = self._manager.field_table.getcol(MS.PHASE_DIR,
            startrow=self._manager.field_id, nrow=1)[0][0]

        return data.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def antenna_position(self, context):
        la, ua = context.dim_extents('na')
        data = self._manager.antenna_table.getcol(MS.POSITION,
                                            startrow=la, nrow=ua-la)
        return data.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def time(self, context):
        lt, ut = context.dim_extents('ntime')
        data = self._manager.ordered_time_table.getcol(MS.TIME,
                                            startrow=lt, nrow=ut-lt)
        return data.astype(context.dtype)

    @cache_ms_read
    def frequency(self, context):
        """ Frequency data source """
        channels = self._manager.spectral_window_table.getcol(MS.CHAN_FREQ)
        return channels.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def ref_frequency(self, context):
        """ Reference frequency data source """
        num_chans = self._manager.spectral_window_table.getcol(MS.NUM_CHAN)
        ref_freqs = self._manager.spectral_window_table.getcol(MS.REF_FREQUENCY)

        data = np.hstack((np.repeat(rf, bs) for bs, rf in zip(num_chans, ref_freqs)))
        return data.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def uvw(self, context):
        """ Per-antenna UVW coordinate data source """
        # Special case for handling antenna uvw code

        # Antenna reading code expects (ntime, nbl) ordering
        if MS.UVW_DIM_ORDER != ('ntime', 'nbl'):
            raise ValueError("'{o}'' ordering expected for "
                "antenna reading code.".format(o=MS.UVW_DIM_ORDER))

        # Figure out our extents in the time dimension
        # and our global antenna and baseline sizes
        (t_low, t_high) = context.dim_extents('ntime')
        na, nbl = context.dim_global_size('na', 'nbl')

        # We expect to handle all antenna at once
        if context.shape != (t_high - t_low, na, 3):
            raise ValueError("Received an unexpected shape "
                "{s} in (ntime,na,3) antenna reading code".format(
                    s=context.shape))

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1.

        # Choosing u_0 = 0 we have:
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2

        # Allocate space for per-antenna UVW, zeroing antenna 0 at each timestep
        ant_uvw = np.empty(shape=context.shape, dtype=context.dtype)
        ant_uvw[:,0,:] = 0

        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(t_low, t_high)):
            # Inspection confirms that this achieves the same effect as
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            self._manager.ordered_uvw_table.getcolnp(MS.UVW,
                ant_uvw[ti,1:na,:],
                startrow=t*nbl, nrow=na-1)

        return ant_uvw

    @cache_ms_read
    def antenna1(self, context):
        """ antenna1 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA1, startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def antenna2(self, context):
        """ antenna2 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA2, startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def observed_vis(self, context):
        """ Observed visibility data source """
        lrow, urow = MS.row_extents(context)

        data = self._manager.ordered_main_table.getcol(
            self._vis_column, startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def flag(self, context):
        """ Flag data source """
        lrow, urow = MS.row_extents(context)

        flag = self._manager.ordered_main_table.getcol(
            MS.FLAG, startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape).astype(context.dtype)

    @cache_ms_read
    def weight(self, context):
        """ Weight data source """
        lrow, urow = MS.row_extents(context)

        weight = self._manager.ordered_main_table.getcol(
            MS.WEIGHT, startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        weight = np.repeat(weight, self._manager.channels_per_band, 0)
        return weight.reshape(context.shape).astype(context.dtype)

    def clear_cache(self):
        """ Clears cached data on this Source Provider """
        self._cache.clear()

    def close(self):
        """ Releases resources associated with this Source Provider """
        self.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__