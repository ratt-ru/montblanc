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

import montblanc

try:
    import pyrap.measures

    pm = pyrap.measures.measures()
except ImportError as e:
    pm = None
    montblanc.log.warning("python-casacore import failed. "
                       "Parallactic Angle computation will fail.")

def parallactic_angles(times, antenna_positions, field_centre, offsets=None):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    Arguments:
        times: ndarray
            Array of unique times with shape (ntime,),
            obtained from TIME column of MS table
        antenna_positions: ndarray of shape (na, 3)
            Antenna positions, obtained from POSITION
            column of MS ANTENNA sub-table
        field_centre : ndarray of shape (2,)
            Field centre, should be obtained from MS PHASE_DIR
        offsets: ndarray of shape (ntime, na, 2) or None
            Containing time-variable offsets in ra, dec from
            pointing centre per antenna. If None is passed offsets
            of zero are assumed.

    Returns:
        An array of parallactic angles per antenna per time-step

    """
    import pyrap.quanta as pq

    try:
        # Create direction measure for the zenith
        zenith = pm.direction('AZEL','0deg','90deg')
    except AttributeError as e:
        if pm is None:
            raise ImportError("python-casacore import failed")

        raise

    # Create position measures for each antenna
    reference_positions = [pm.position('itrf',
        *(pq.quantity(x,'m') for x in pos))
        for pos in antenna_positions]

    # Compute pointing centre in radians
    na = antenna_positions.shape[0]
    nt = times.shape[0]
    if offsets is None:
        offsets = np.zeros((nt, na, 2))

    fc_rad = np.asarray([[pm.direction('J2000',
                            pq.quantity(field_centre[0] + offsets[t, a, 0], 'rad'),
                            pq.quantity(field_centre[1] + offsets[t, a, 1], 'rad'))
                          for a in range(na)]
                         for t in range(nt)])

    return np.asarray([
            # Set current time as the reference frame
            pm.do_frame(pm.epoch("UTC", pq.quantity(t, "s")))
            and
            [   # Set antenna position as the reference frame
                pm.do_frame(rp)
                and
                pm.posangle(fc_rad[ti, ai], zenith).get_value("rad")
                for ai, rp in enumerate(reference_positions)
            ]
        for ti, t in enumerate(times)])
