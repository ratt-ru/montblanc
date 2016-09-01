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

import montblanc

from montblanc.impl.rime.tensorflow.sinks.rime_data_sink import RimeDataSink
import montblanc.impl.rime.tensorflow.ms.ms_manager as MS

class MSRimeDataSink(RimeDataSink):
    def __init__(self, manager):
        self._manager = manager

    def model_vis(self, context):
        lrow, urow = MS.row_extents(context)

        column = 'MODEL_DATA'
        colshape = self._manager.column_descriptors[column]['shape']
        msshape = [-1] + colshape.tolist()

        self._manager.ordered_main_table.putcol(column,
            context.data.reshape(msshape),
            startrow=lrow, nrow=urow-lrow)



