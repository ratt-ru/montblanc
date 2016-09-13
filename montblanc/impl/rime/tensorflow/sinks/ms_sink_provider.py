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

from montblanc.config import RimeSolverConfig as Options
from montblanc.impl.rime.tensorflow.sinks.sink_provider import SinkProvider
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

class MSSinkProvider(SinkProvider):
    def __init__(self, manager):
        self._manager = manager
        self._name = "Measurement Set '{ms}'".format(ms=manager.msname)

    def name(self):
        return self._name

    def model_vis(self, context):
        lrow, urow = MS.row_extents(context)

        column = context.cfg[Options.MS_VIS_OUTPUT_COLUMN]

        try:
            coldesc = self._manager.column_descriptors[column]
        except KeyError as e:
            raise (ValueError("No column descriptor for '{c}'".format(c=column)),
                None, sys.exc_info()[2])

        msshape = [-1] + coldesc['shape'].tolist()

        self._manager.ordered_main_table.putcol(column,
            context.data.reshape(msshape),
            startrow=lrow, nrow=urow-lrow)


    def __str__(self):
        return self.__class__.__name__

