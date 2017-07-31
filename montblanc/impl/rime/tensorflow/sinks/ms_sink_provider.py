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

import sys

import montblanc

from montblanc.impl.rime.tensorflow.sinks.sink_provider import SinkProvider
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

class MSSinkProvider(SinkProvider):
    """
    Sink Provider that receives model visibilities produced by
    montblanc
    """

    def __init__(self, manager, vis_column=None):
        """
        Constructs an MSSinkProvider object

        Parameters
        ----------
        manager: :py:class:`.MeasurementSetManager`
            The :py:class:`.MeasurementSetManager` used to access
            the Measurement Set.
        vis_column: str
            Column to which model visibilities will be read
        """

        self._manager = manager
        self._name = "Measurement Set '{ms}'".format(ms=manager.msname)
        self._vis_column = ('CORRECTED_DATA' if vis_column is None else vis_column)

    def name(self):
        return self._name

    def model_vis(self, context):
        """ model visibility data sink """
        column = self._vis_column
        msshape = None

        # Do we have a column descriptor for the supplied column?
        try:
            coldesc = self._manager.column_descriptors[column]
        except KeyError as e:
            coldesc = None

        # Try to get the shape from the descriptor
        if coldesc is not None:
            try:
                msshape = [-1] + coldesc['shape'].tolist()
            except KeyError as e:
                msshape = None

        # Otherwise guess it and warn
        if msshape is None:
            guessed_shape = [self._manager._nchan, 4]

            montblanc.log.warn("Could not obtain 'shape' from the '{c}' "
                "column descriptor. Guessing it is '{gs}'.".format(
                    c=column, gs=guessed_shape))

            msshape = [-1] + guessed_shape

        lrow, urow = MS.row_extents(context)

        self._manager.ordered_main_table.putcol(column,
            context.data.reshape(msshape),
            startrow=lrow, nrow=urow-lrow)

    def __str__(self):
        return self.__class__.__name__

