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

from montblanc.impl.rime.tensorflow.sinks.sink_provider import SinkProvider
from montblanc.impl.rime.tensorflow.sinks.null_sink_provider import NullSinkProvider
from montblanc.impl.rime.tensorflow.sinks.ms_sink_provider import MSSinkProvider
from montblanc.impl.rime.tensorflow.sinks.sink_context import SinkContext