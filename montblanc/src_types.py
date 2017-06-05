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

from collections import OrderedDict

import numpy as np

import montblanc

# List of source types and the variable names
# referring to the number of sources for that type
POINT_TYPE = 'point'
POINT_NR_VAR = 'npsrc'

GAUSSIAN_TYPE = 'gaussian'
GAUSSIAN_NR_VAR = 'ngsrc'

SERSIC_TYPE = 'sersic'
SERSIC_NR_VAR = 'nssrc'

# Type to numbering variable mapping,
# with a specific ordering (point first)
SOURCE_VAR_TYPES = OrderedDict([
    (POINT_TYPE, POINT_NR_VAR),
    (GAUSSIAN_TYPE, GAUSSIAN_NR_VAR),
    (SERSIC_TYPE, SERSIC_NR_VAR)
])

SOURCE_DIM_TYPES = OrderedDict((v, k) for (k, v)
    in SOURCE_VAR_TYPES.iteritems())

def source_types():
    """ Returns a list of registered source types """
    return SOURCE_VAR_TYPES.keys()

def source_nr_vars():
    """ Returns a list of registered source number variables """
    return SOURCE_VAR_TYPES.values()

def source_var_types():
    """ Returns a mapping of source type to number variable """
    return SOURCE_VAR_TYPES

def default_sources(**kwargs):
    """
    Returns a dictionary mapping source types
    to number of sources. If the number of sources
    for the source type is supplied in the kwargs
    these will be placed in the dictionary.

    e.g. if we have 'point', 'gaussian' and 'sersic'
    source types, then

    default_sources(point=10, gaussian=20)

    will return an OrderedDict {'point': 10, 'gaussian': 20, 'sersic': 0}
    """
    S = OrderedDict()
    total = 0

    invalid_types = [t for t in kwargs.keys() if t not in SOURCE_VAR_TYPES]

    for t in invalid_types:
        montblanc.log.warning('Source type %s is not yet '
            'implemented in montblanc. '
            'Valid source types are %s' % (t, SOURCE_VAR_TYPES.keys()))

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

    will return an OrderedDict

    {'npsrc': 10, 'ngsrc': 20, 'nssrc': 0 }
    """

    sources = default_sources(**sources)

    try:
        return OrderedDict((SOURCE_VAR_TYPES[name], nr)
            for name, nr in sources.iteritems())
    except KeyError as e:
        raise KeyError((
            'No source type ''%s'' is '
            'registered. Valid source types '
            'are %s') % (e, SOURCE_VAR_TYPES.keys()))

def source_range_tuple(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing tuples of the start and end index
    for each source variable type.
    """

    starts = np.array([0 for nr_var in SOURCE_VAR_TYPES.itervalues()])
    ends = np.array([nr_var_dict[nr_var] if nr_var in nr_var_dict else 0
        for nr_var in SOURCE_VAR_TYPES.itervalues()])
    sum_counts = np.cumsum(ends)
    idx = np.arange(len(starts))

    # Find the intervals containing the
    # start and ending indices
    start_idx, end_idx = np.searchsorted(
        sum_counts, [start, end], side='right')

    # Handle edge cases
    if end >= sum_counts[-1]:
        end = sum_counts[-1]
        end_idx = len(sum_counts) - 1

    # Find out which variable counts fall within the range
    # of the supplied indices and zero those outside this range
    invalid = np.logical_not(np.logical_and(start_idx <= idx, idx <= end_idx))
    starts[invalid] = ends[invalid] = 0

    # Modify the associated starting and ending positions
    starts[start_idx] = start
    ends[end_idx] = end

    if start >= sum_counts[0]:
        starts[start_idx] -= sum_counts[start_idx-1]

    if end >= sum_counts[0]:
        ends[end_idx] -= sum_counts[end_idx-1]

    return OrderedDict((n, (starts[i], ends[i]))
        for i, n in enumerate(SOURCE_VAR_TYPES.values()))

def source_range(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing tuples of the start and end index
    for each source variable type.
    """

    return OrderedDict((k, e-s)
        for k, (s, e)
        in source_range_tuple(start, end, nr_var_dict).iteritems())

def source_range_slices(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing slices for each source variable type.
    """

    return OrderedDict((k, slice(s,e,1))
        for k, (s, e)
        in source_range_tuple(start, end, nr_var_dict).iteritems())