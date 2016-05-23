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

def default_ant_pairs(self):
    """
    Return a list of 2 arrays of shape (ntime, nbl)
    containing the default antenna pairs for each timestep
    at each baseline.
    """

    # Create the antenna pair mapping, from upper triangle indices
    # based on the number of antenna.
    ntime, nbl = self.dim_local_size('ntime', 'nbl')
    ant0, ant1 = self.default_base_ant_pairs()

    return (np.tile(ant0, ntime).reshape(ntime, nbl),
        np.tile(ant1, ntime).reshape(ntime, nbl))

def ap_idx(self, default_ap=None, src=False, chan=False):
    """
    This method produces a pair of indices
    which arranges per antenna values into a
    per baseline configuration, using the supplied default_ap
    per timestep and baseline antenna pair configuration.
    Thus, indexing an array with shape (na) will produce
    a view of the values in this array with shape (nbl).

    Consequently, this method is suitable for indexing
    an array of shape (ntime, na). Specifiying source
    and channel dimensions allows indexing of an array
    of shape (nsrc, ntime, na, nchan).

    Index an array of (ntime, na) with one of the 
    returned indices produces a (ntime, nbl) array,
    or (nsrc, ntime, nbl, nchan) if source
    and channel are also included.

    The values for antenna one and two are in the
    first and second indices, respectively.

    >>> ant0, ant1 = slvr.ap_idx()
    >>> u_ant = np.random.random(size=(ntime,na))
    >>> u_bl = u_ant[ant1] - u_ant[ant0]
    >>> assert u_bl.shape == (ntime, nbl)
    """

    slvr = self

    if default_ap is None:
        default_ap = self.default_ant_pairs()

    ant0, ant1 = default_ap
    idx0, idx1 = [], []

    needed = (src, True, True, chan)
    nsrc, ntime, nbl, nchan = slvr.dim_local_size(
        'nsrc', 'ntime', 'nbl', 'nchan')

    if src:
        src_shape = tuple(s for s,n in zip((nsrc,1,1,1), needed) if n)
        src_range = np.arange(nsrc).reshape(src_shape)
        idx0.append(src_range)
        idx1.append(src_range)

    time_shape = tuple(t for t, n in zip((1,ntime,1,1), needed) if n)
    time_range = np.arange(ntime).reshape(time_shape)
    idx0.append(time_range)
    idx1.append(time_range)

    ant_shape = tuple(a for a, n in zip((1,ntime,nbl,1), needed) if n)
    idx0.append(ant0.reshape(ant_shape))
    idx1.append(ant1.reshape(ant_shape))

    if chan:
        chan_shape = tuple(c for c, n in zip((1,1,1,nchan), needed) if n)
        chan_range = np.arange(nchan).reshape(chan_shape)
        idx0.append(chan_range)
        idx1.append(chan_range)

    return idx0, idx1

def monkey_patch_antenna_pairs(slvr):
    # Monkey patch these functions onto the solver object
    import types

    slvr.default_ant_pairs = types.MethodType(
        default_ant_pairs, slvr)

    slvr.ap_idx = types.MethodType(
        ap_idx, slvr)