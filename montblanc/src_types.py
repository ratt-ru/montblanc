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

POINT_TYPE = 'point'
GAUSSIAN_TYPE = 'gaussian'
SERSIC_TYPE = 'sersic'

POINT_NR_VAR = 'npsrc'
GAUSSIAN_NR_VAR = 'ngsrc'
SERSIC_NR_VAR = 'nssrc'

#SOURCE_TYPES = [POINT_TYPE, GAUSSIAN_TYPE, SERSIC_TYPE]
#SOURCE_NR_VARS = [POINT_NR_VAR, GAUSSIAN_NR_VAR, SERSIC_NR_VAR]

SOURCES_TYPES = [
    { 'name': POINT_TYPE, 'nr_var': POINT_NR_VAR },
    { 'name': GAUSSIAN_TYPE, 'nr_var': GAUSSIAN_NR_VAR },
    { 'name': SERSIC_TYPE, 'nr_var': SERSIC_NR_VAR },
]

def default_src_cfg():
    return {}