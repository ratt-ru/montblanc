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

from sink_provider import SinkProvider

class NullSinkProvider(SinkProvider):

    def name(self):
        return "Null"

    def model_vis(self, context):
        array_schema = context.array(context.name)
        slices = context.slice_index(*array_schema.shape)
        slice_str = ','.join('%s:%s' % (s.start, s.stop) for s in slices)
        montblanc.log.info("Received '{n}[{sl}]"
            .format(n=context.name, sl=slice_str))

    def chi_squared(self, context):
        array_schema = context.array(context.name)
        slices = context.slice_index(*array_schema.shape)
        slice_str = ','.join('%s:%s' % (s.start, s.stop) for s in slices)
        montblanc.log.info("Received '{n}[{sl}]"
            .format(n=context.name, sl=slice_str))

    def __str__(self):
        return self.__class__.__name__