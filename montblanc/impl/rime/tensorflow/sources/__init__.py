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

from montblanc.impl.rime.tensorflow.sources.source_context import SourceContext
from montblanc.impl.rime.tensorflow.sources.source_provider import (SourceProvider,
    find_sources)
from montblanc.impl.rime.tensorflow.sources.ms_source_provider import MSSourceProvider
from montblanc.impl.rime.tensorflow.sources.np_source_provider import NumpySourceProvider
from montblanc.impl.rime.tensorflow.sources.fits_beam_source_provider import FitsBeamSourceProvider