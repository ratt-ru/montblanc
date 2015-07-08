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

import numpy as np

POINT_TYPE = 'point'
GAUSSIAN_TYPE = 'gaussian'
SERSIC_TYPE = 'sersic'

POINT_NR_VAR = 'npsrc'
GAUSSIAN_NR_VAR = 'ngsrc'
SERSIC_NR_VAR = 'nssrc'

SOURCE_VAR_TYPES = {
    POINT_TYPE : POINT_NR_VAR,
    GAUSSIAN_TYPE : GAUSSIAN_NR_VAR,
    SERSIC_TYPE : SERSIC_NR_VAR
}

def default_sources(**kwargs):
    """
    Returns a dictionary mapping source types
    to number of sources. If the number of sources
    for the source type is supplied in the kwargs
    these will be placed in the dictionary.

    e.g. if we have 'point', 'gaussian' and 'sersic'
    source types, then

    default_sources(point=10, gaussian=20)

    will return a dict {'point': 10, 'gaussian': 20, 'sersic': 0}
    """
    S = {}
    total = 0

    # Zero all source types
    for k, v in SOURCE_VAR_TYPES.iteritems():
        # Try get the number of sources for this source
        # from the kwargs
        value = kwargs.get(k, 0)

        try:
            value = int(value)
        except ValueError:
            raise TypeError(('Supplied value %s '
                'for source %s cannot be '
                'converted to an integer') % \
                    (value, k))    

        total += value
        S[k] = value

    # Add a point source if no others exist
    if total == 0:
        S[POINT_TYPE] = 1

    return S

def sources_to_nr_vars(sources):
    """
    Converts a source type to number of sources mapping into
    a source numbering variable to number of sources mapping.

    If, for example, we have 'point', 'gaussian' and 'sersic'
    source types, then passing the following dict as an argument
    
    sources_to_nr_vars({'point':10, 'gaussian': 20})
    
    will return a new dict

    {'npsrc': 10, 'ngsrc': 20, 'nssrc': 0 }
    """

    sources = default_sources(**sources)

    try:
        return { SOURCE_VAR_TYPES[name]: nr for name, nr in sources.iteritems() }
    except KeyError as e:
        raise KeyError((
            'No source type ''%s'' is '
            'registered. Valid source types '
            'are %s') % (e, SOURCE_VAR_TYPES.keys()))