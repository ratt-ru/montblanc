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
from montblanc.impl.rime.tensorflow.sources import SourceContext

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

        # Hacky access of private member
        cube = context._cube

        # Create antenna1 source context
        a1_actual = cube.array("antenna1", reify=True)
        a1_ctx = SourceContext("antenna1", cube, context.cfg,
            context.iter_args, cube.array("antenna1"),
            a1_actual.shape, a1_actual.dtype)

        # Create antenna2 source context
        a2_actual = cube.array("antenna2", reify=True)
        a2_ctx = SourceContext("antenna2", cube, context.cfg,
            context.iter_args, cube.array("antenna2"),
            a2_actual.shape, a2_actual.dtype)

        # Get antenna1 and antenna2 data
        ant1 = self.antenna1(a1_ctx).ravel()
        ant2 = self.antenna2(a2_ctx).ravel()

        # Obtain per baseline UVW data
        lrow, urow = MS.uvw_row_extents(context)
        uvw = self._manager.ordered_uvw_table.getcol(MS.UVW,
                                                startrow=lrow,
                                                nrow=urow-lrow)

        # Perform the per-antenna UVW decomposition
        ntime, nbl = context.dim_extent_size('ntime', 'nbl')
        na = context.dim_global_size('na')
        chunks = np.repeat(nbl, ntime).astype(ant1.dtype)

        auvw = mbu.antenna_uvw(uvw, ant1, ant2, chunks, nr_of_antenna=na)

        return auvw.reshape(context.shape).astype(context.dtype)

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
