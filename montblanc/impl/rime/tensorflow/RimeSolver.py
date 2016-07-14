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
import montblanc.util as mbu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

class RimeSolver(MontblancTensorflowSolver):
    """ RIME Solver Implementation """

    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        super(RimeSolver, self).__init__(slvr_cfg)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube nu depth')

        # Monkey patch these functions onto the object
        from montblanc.impl.rime.tensorflow.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)
   
        from montblanc.impl.rime.tensorflow.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        # Find out which dimensions have been modified by budgeting
        # and update them
        modded_dims = self._budget(A, slvr_cfg)

        for k, v in modded_dims.iteritems():
            self.update_dimension(k, local_size=v,
                lower_extent=0, upper_extent=v)

        from montblanc.impl.rime.tensorflow.feeders.queue_wrapper import create_queue_wrapper

        # Get the data source (defaults or test data)
        data_source = slvr_cfg.get(Options.DATA_SOURCE)

        # Set up the queue data sources. Just take from
        # the defaults if the original data source was MS
        # we only want the data source types for configuring
        # the queue
        queue_data_source = (Options.DATA_SOURCE_DEFAULT
            if data_source == Options.DATA_SOURCE_MS
            else data_source)

        montblanc.log.info("Taking queue defaults from data source '{ds}'"
            .format(ds=queue_data_source))

        # Obtain default data sources for each array,
        # then update with any data sources supplied by the user
        ds = { n: (a.get(queue_data_source), a.dtype)
            for n, a in self.arrays().iteritems() }
        ds.update(slvr_cfg.get('supplied', {}))

        # Test data sources here
        ary_descs = self.arrays(reify=True)

        for n, (s, t) in ds.iteritems():
            print 'Testing source {n} with shape {s}'.format(n=n, s=ary_descs[n].shape)
            if s is not None:
                a = s(self, ary_descs[n])
                print a.flatten()[0:10]

        QUEUE_SIZE = 10

        self._uvw_queue = create_queue_wrapper(QUEUE_SIZE,
            ['uvw', 'antenna1', 'antenna2'], ds)

        self._observation_queue = create_queue_wrapper(QUEUE_SIZE,
            ['observed_vis', 'flag', 'weight'], ds)

        self._die_queue = create_queue_wrapper(QUEUE_SIZE,
            ['gterm'], ds)

        self._dde_queue = create_queue_wrapper(QUEUE_SIZE,
            ['ebeam', 'antenna_scaling', 'point_errors'], ds)

        self._output_queue = create_queue_wrapper(QUEUE_SIZE,
            ['model_vis'], ds)

    def _budget(self, arrays, slvr_cfg):
        na = slvr_cfg.get(Options.NA)
        nsrc = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)
        src_str_list = [Options.NSRC] + mbu.source_nr_vars()
        src_reduction_str = '&'.join(['%s=%s' % (nr_var, nsrc)
            for nr_var in src_str_list])

        mem__budget = slvr_cfg.get('mem_budget', 256*ONE_MB)
        T = self.template_dict()

        # Figure out a viable dimension configuration
        # given the total problem size 
        viable, modded_dims = mbu.viable_dim_config(
            mem__budget, arrays, T, [src_reduction_str,
                'ntime',
                'nbl={na}&na={na}'.format(na=na)], 1)                

        # Create property dictionary with updated dimensions.
        # Determine memory required by our chunk size
        mT = T.copy()
        mT.update(modded_dims)
        required_mem = mbu.dict_array_bytes_required(arrays, mT)

        # Log some information about the memory _budget
        # and dimension reduction
        montblanc.log.info(("Selected a solver memory _budget of {rb} "
            "given a hard limit of {mb}.").format(
            rb=mbu.fmt_bytes(required_mem),
            mb=mbu.fmt_bytes(mem__budget)))

        montblanc.log.info((
            "The following dimension reductions "
            "have been applied:"))

        for k, v in modded_dims.iteritems():
            montblanc.log.info('{p}{d}: {id} => {rd}'.format
                (p=' '*4, d=k, id=T[k], rd=v))

        return modded_dims

    def solve(self):
        pass