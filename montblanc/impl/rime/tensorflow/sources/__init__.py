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

def test():
    import argparse
    import copy
    import time

    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('msfile')
    args = parser.parse_args()

    source = MSSourceProvider(args.msfile, precision='float')
    cube = copy.deepcopy(source.mscube)

    row_iter_sizes = [10] + cube.dim_global_size('nbl', 'nbands')
    dim_iter_args = zip(source.MS_DIM_ORDER, row_iter_sizes)

    # Arrays that we will feed
    array_names = ('antenna1', 'antenna2', 'uvw',
            'observed_vis', 'flag', 'weight', 'parallactic_angles')

    def feed(source, cube, array_names):
        for dims in cube.dim_iter(*dim_iter_args, update_local_size=True):
            cube.update_dimensions(dims)
            cube.update_row_dimensions()
            array_schemas = cube.arrays(reify=True)

            source_contexts = ((n, SourceContext(n, cube, {},
                array_schemas[n].shape, array_schemas[n].dtype))
                for n in array_names)

            feed_arrays = ((n, getattr(source, n)(c)) for n, c in source_contexts)

            print ' '.join(['{n} {s}'.format(n=n,s=a.shape) for n, a in feed_arrays])

    start = time.clock()
    feed(source, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    #source.clear_cache()

    start = time.clock()
    feed(source, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    array_names = ('antenna1', 'antenna2', 'uvw',
            'observed_vis', 'flag', 'weight', 'parallactic_angles')

    cube = copy.deepcopy(source.mscube)
    array_schemas = cube.arrays(reify=True)
    arrays = { a: np.zeros(s.shape, s.dtype) for (a, s) in
        ((a, array_schemas[a]) for a in array_names) }

    print [(k, a.shape) for k, a in arrays.iteritems()]

    source = NumpySourceProvider(arrays, cube)
    feed(source, cube, array_names)

if __name__ == '__main__':
    test()
