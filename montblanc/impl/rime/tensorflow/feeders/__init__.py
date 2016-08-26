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

from montblanc.impl.rime.tensorflow.feeders.feed_context import FeedContext
from montblanc.impl.rime.tensorflow.feeders.rime_data_feeder import RimeDataFeeder
from montblanc.impl.rime.tensorflow.feeders.ms_data_feeder import MSRimeDataFeeder
from montblanc.impl.rime.tensorflow.feeders.np_data_feeder import NumpyRimeDataFeeder
from montblanc.impl.rime.tensorflow.feeders.fits_beam_data_feeder import FitsBeamDataFeeder

def test():
    import argparse
    import copy
    import time
   
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('msfile')
    args = parser.parse_args()

    feeder = MSRimeDataFeeder(args.msfile, precision='float')
    cube = copy.deepcopy(feeder.mscube)

    row_iter_sizes = [10] + cube.dim_global_size('nbl', 'nbands')
    dim_iter_args = zip(feeder.MS_DIM_ORDER, row_iter_sizes)

    # Arrays that we will feed
    array_names = ('antenna1', 'antenna2', 'uvw',
            'observed_vis', 'flag', 'weight', 'parallactic_angles')

    def feed(feeder, cube, array_names):
        for dims in cube.dim_iter(*dim_iter_args, update_local_size=True):
            cube.update_dimensions(dims)
            cube.update_row_dimensions()
            array_schemas = cube.arrays(reify=True)

            feed_contexts = ((n, FeedContext(n, cube, {},
                array_schemas[n].shape, array_schemas[n].dtype))
                for n in array_names)

            feed_arrays = ((n, getattr(feeder, n)(c)) for n, c in feed_contexts)

            print ' '.join(['{n} {s}'.format(n=n,s=a.shape) for n, a in feed_arrays])

    start = time.clock()
    feed(feeder, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    #feeder.clear_cache()

    start = time.clock()
    feed(feeder, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    array_names = ('antenna1', 'antenna2', 'uvw',
            'observed_vis', 'flag', 'weight', 'parallactic_angles')

    cube = copy.deepcopy(feeder.mscube)
    array_schemas = cube.arrays(reify=True)
    arrays = { a: np.zeros(s.shape, s.dtype) for (a, s) in
        ((a, array_schemas[a]) for a in array_names) }

    print [(k, a.shape) for k, a in arrays.iteritems()]

    feeder = NumpyRimeDataFeeder(arrays, cube)
    feed(feeder, cube, array_names)

if __name__ == '__main__':
    test()
