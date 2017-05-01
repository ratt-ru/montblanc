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

import threading

import numpy as np

class MeasuresServerWrapper(object):
    """
    Thread-safe Context Manager for a measures server object

    .. code-block:: python

        server = MeasuresServerWrapper()

        with server as pm:
            zenith = pm.direction('AZEL','0deg','90deg')
    """
    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()

        try:
            # Already created, re-use
            return self._measure
        except AttributeError:
            # Create
            import pyrap.measures
            self._measure = pyrap.measures.measures()

        return self._measure

    def __exit__(self, evalue, etype, etraceback):
        self._lock.release()

_measures_server = MeasuresServerWrapper()

def parallactic_angles(times, antenna_positions, field_centre):
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

    Returns:
        An array of parallactic angles per time-step

    """
    import pyrap.quanta as pq

    with _measures_server as pm:
        # Create direction measure for the zenith
        zenith = pm.direction('AZEL','0deg','90deg')

        # Create position measures for each antenna
        reference_positions = [pm.position('itrf',
            *(pq.quantity(x,'m') for x in pos))
            for pos in antenna_positions]

        # Compute field centre in radians
        fc_rad = pm.direction('J2000',
            *(pq.quantity(f,'rad') for f in field_centre))

        parallactic_angles = np.asarray([[
                # Set antenna position as the reference frame
                pm.do_frame(rp) and
                # Set current time as the reference frame
                pm.do_frame(pm.epoch("UTC",pq.quantity(t,"s"))) and
                # Now compute the parallactic angle
                pm.posangle(fc_rad, zenith).get_value("rad")]
            for t in times
            for rp in reference_positions])

        return parallactic_angles
