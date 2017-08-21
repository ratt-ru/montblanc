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
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

from montblanc.impl.rime.tensorflow.sources.source_provider import SourceProvider

class MSSourceProvider(SourceProvider):
    """
    Source Provider that retrieves input data from a
    MeasurementSet
    """

    def __init__(self, manager, vis_column=None):
        """
        Constructs an MSSourceProvider object

        Parameters
        ----------
        manager: :py:class:`.MeasurementSetManager`
            The :py:class:`.MeasurementSetManager` used to access
            the Measurement Set.
        vis_column: str
            Column from which observed visibilities will be read
        """
        self._manager = manager
        self._name = "Measurement Set '{ms}'".format(ms=manager.msname)

        self._vis_column = 'DATA' if vis_column is None else vis_column

        # Cache columns on the object
        # Handle these columns slightly differently
        # They're used to compute the parallactic angle
        # TODO: Fit them into the cache_ms_read strategy at some point

        # Cache antenna positions
        self._antenna_positions = manager.antenna_table.getcol(MS.POSITION)

        # Cache timesteps
        self._times = manager.ordered_time_table.getcol(MS.TIME)

        # Cache the phase direction for the field
        # [0][0] because (a) we select only 1 row
        #                (b) assumes a NUM_POLY of 1
        self._phase_dir = manager.field_table.getcol(MS.PHASE_DIR,
            startrow=manager.field_id, nrow=1)[0][0]

    def name(self):
        return self._name

    def updated_dimensions(self):
        # Defer to manager's method
        return self._manager.updated_dimensions()

    def phase_centre(self, context):
        return self._phase_dir.astype(context.dtype)

    def antenna_position(self, context):
        la, ua = context.dim_extents('na')
        return (self._antenna_positions[la:ua]
                    .astype(context.dtype))

    def time(self, context):
        lt, ut = context.dim_extents('ntime')
        return self._times[lt:ut].astype(context.dtype)

    def frequency(self, context):
        """ Frequency data source """
        channels = self._manager.spectral_window_table.getcol(MS.CHAN_FREQ)
        return channels.reshape(context.shape).astype(context.dtype)

    def ref_frequency(self, context):
        """ Reference frequency data source """
        num_chans = self._manager.spectral_window_table.getcol(MS.NUM_CHAN)
        ref_freqs = self._manager.spectral_window_table.getcol(MS.REF_FREQUENCY)

        data = np.hstack((np.repeat(rf, bs) for bs, rf in zip(num_chans, ref_freqs)))
        return data.reshape(context.shape).astype(context.dtype)

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

    def antenna1(self, context):
        """ antenna1 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna1 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA1, startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape).astype(context.dtype)

    def antenna2(self, context):
        """ antenna2 data source """
        lrow, urow = MS.uvw_row_extents(context)
        antenna2 = self._manager.ordered_uvw_table.getcol(
            MS.ANTENNA2, startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape).astype(context.dtype)

    def parallactic_angles(self, context):
        """ parallactic angle data source """
        # Time and antenna extents
        (lt, ut), (la, ua) = context.dim_extents('ntime', 'na')

        return (mbu.parallactic_angles(self._times[lt:ut],
                self._antenna_positions[la:ua], self._phase_dir)
                                            .reshape(context.shape)
                                            .astype(context.dtype))


    def observed_vis(self, context):
        """ Observed visibility data source """
        lrow, urow = MS.row_extents(context)

        data = self._manager.ordered_main_table.getcol(
            self._vis_column, startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape).astype(context.dtype)

    def flag(self, context):
        """ Flag data source """
        lrow, urow = MS.row_extents(context)

        flag = self._manager.ordered_main_table.getcol(
            MS.FLAG, startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape).astype(context.dtype)

    def weight(self, context):
        """ Weight data source """
        lrow, urow = MS.row_extents(context)

        weight = self._manager.ordered_main_table.getcol(
            MS.WEIGHT, startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        weight = np.repeat(weight, self._manager.channels_per_band, 0)
        return weight.reshape(context.shape).astype(context.dtype)

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

    def __str__(self):
        return self.__class__.__name__