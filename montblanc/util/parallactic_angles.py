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
    montblanc.log.warn("python-casacore import failed. "
                       "Parallactic Angle computation will fail.")

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

    try:
        # Create direction measure for the zenith
        zenith = pm.direction('AZELGEO','0deg','90deg')
    except AttributeError as e:
        if pm is None:
            raise ImportError("python-casacore import failed")

        raise

    # Create position measures for each antenna
    reference_positions = [pm.position('itrf',
        *(pq.quantity(x,'m') for x in pos))
        for pos in antenna_positions]

    # Compute field centre in radians
    fc_rad = pm.direction('J2000',
        *(pq.quantity(f,'rad') for f in field_centre))

    return np.asarray([
            # Set current time as the reference frame
            pm.do_frame(pm.epoch("UTC", pq.quantity(t, "s")))
            and
            [   # Set antenna position as the reference frame
                pm.do_frame(rp)
                and
                pm.posangle(fc_rad, zenith).get_value("rad")
                for rp in reference_positions
            ]
        for t in times])


from astropy.coordinates import EarthLocation, SkyCoord, AltAz, CIRS, Angle
from astropy.time import Time

from astropy import units

def _parallactic_angle_astropy(times, ap, fc):
    """
    Computes parallactic angles per timestep for the given
    reference antenna position and field centre.

    Arguments:
        times: ndarray
            Array of unique times with shape (ntime,),
            obtained from TIME column of MS table
        ap: ndarray of shape (na, 3)
            Antenna positions, obtained from POSITION
            column of MS ANTENNA sub-table
        fc : ndarray of shape (2,)
            Field centre, should be obtained from MS PHASE_DIR

    Returns:
        An array of parallactic angles per time-step

    """
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, CIRS
    from astropy.time import Time
    from astropy import units

    # Convert from MJD second to MJD
    times = Time(times / 86400.00, format='mjd', scale='utc')

    ap = EarthLocation.from_geocentric(ap[:,0], ap[:,1], ap[:,2], unit='m')
    fc = SkyCoord(ra=fc[0], dec=fc[1], unit=units.rad, frame='fk5')
    pole = SkyCoord(ra=0, dec=90, unit=units.deg, frame='fk5')

    altaz_frame = AltAz(location=ap[None,:], obstime=times[:,None])
    pole_altaz = pole.transform_to(altaz_frame)
    fc_altaz = fc.transform_to(altaz_frame)
    return fc_altaz.position_angle(pole_altaz)

if __name__ == "__main__":
    # 5s second types from 12h00 midday on 1st Feb
    times = np.arange('2017-02-01T12:00', '2017-02-01T16:00', dtype='datetime64[5s]')
    ftimes = times.astype('datetime64[s]').astype(np.float64)

    # Westerbork antenna positions
    antenna_positions = np.array([
           [ 3828763.10544699,   442449.10566454,  5064923.00777   ],
           [ 3828746.54957258,   442592.13950824,  5064923.00792   ],
           [ 3828729.99081359,   442735.17696417,  5064923.00829   ],
           [ 3828713.43109885,   442878.2118934 ,  5064923.00436   ],
           [ 3828696.86994428,   443021.24917264,  5064923.00397   ],
           [ 3828680.31391933,   443164.28596862,  5064923.00035   ],
           [ 3828663.75159173,   443307.32138056,  5064923.00204   ],
           [ 3828647.19342757,   443450.35604638,  5064923.0023    ],
           [ 3828630.63486201,   443593.39226634,  5064922.99755   ],
           [ 3828614.07606798,   443736.42941621,  5064923.        ],
           [ 3828609.94224429,   443772.19450029,  5064922.99868   ],
           [ 3828601.66208572,   443843.71178407,  5064922.99963   ],
           [ 3828460.92418735,   445059.52053929,  5064922.99071   ],
           [ 3828452.64716351,   445131.03744105,  5064922.98793   ]],
                dtype=np.float64)

    phase_centre = np.array([ 0.        ,  1.04719755], dtype=np.float64)

    import time

    t = time.time()
    pa_astro = _parallactic_angle_astropy(ftimes, antenna_positions, phase_centre)
    print "pa_astropy done in ", time.time() - t

    t = time.time()
    pa_casa = parallactic_angles(ftimes, antenna_positions, phase_centre)
    print "pa_casa done in ", time.time() - t

    pa_astro = Angle(pa_astro, unit=units.deg).wrap_at(180*units.deg)
    pa_casa = Angle(pa_casa*units.rad, unit=units.deg).wrap_at(180*units.deg)

    for a, c in zip(pa_astro.flat, pa_casa.flat):
        print a, c, (a-c).wrap_at(180*units.deg)



