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

import collections
import itertools
import os

from attrdict import AttrDict
import concurrent.futures as cf
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow.ant_pairs import monkey_patch_antenna_pairs
from montblanc.impl.rime.tensorflow.cube_dim_transcoder import CubeDimensionTranscoder
from montblanc.impl.rime.tensorflow.feeders.feed_context import FeedContext

from hypercube import HyperCube
import hypercube.util as hcu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

rime_lib_path = os.path.join(montblanc.get_montblanc_path(),
    'tensorflow', 'rime_ops', 'rime.so')
rime = tf.load_op_library(rime_lib_path)

DataSource = collections.namedtuple("DataSource", ['source', 'type'])

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

        #=========================================
        # Register hypercube Dimensions
        #=========================================

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
        monkey_patch_antenna_pairs(self)

        #=========================================
        # Register hypercube Arrays and Properties
        #=========================================
   
        from montblanc.impl.rime.tensorflow.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        #==================
        # Memory Budgeting
        #==================

        # Attempt to fit arrays into memory budget by
        # reducing dimension local_sizes
        modded_dims = self._budget(A, slvr_cfg)

        # Update any dimensions
        for k, v in modded_dims.iteritems():
            self.update_dimension(k, local_size=v,
                lower_extent=0, upper_extent=v)

        #====================================
        # Tensorflow Session and placeholders
        #====================================

        # Create the tensorflow session object
        self._tf_session = tf.Session()

        # Create placholder variables for source counts
        self._src_ph_vars = AttrDict({
            n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
            for n in mbu.source_nr_vars() })

        # Create placeholder variables for properties
        self._property_ph_vars = AttrDict({
            n: tf.placeholder(dtype=p.dtype, shape=(), name=n)
            for n, p in self.properties().iteritems() })

        #================================
        # Queue Data Source Configuration
        #================================

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
        self._data_sources = ds = {
            n: DataSource(a.get(queue_data_source), a.dtype)
            for n, a in self.arrays().iteritems() }

        ds.update(slvr_cfg.get('supplied', {}))

        # The descriptor queue items are not user-defined arrays
        # but a variable passed through describing a chunk of the
        # problem. Make it look as if it's an array
        if 'descriptor' in ds:
            raise KeyError("'descriptor' is reserved, "
                "please use another array name.")

        ds['descriptor'] = DataSource(lambda c: np.int32([0]), np.int32)

        QUEUE_SIZE = 10

        from montblanc.impl.rime.tensorflow.feeders.queue_wrapper import create_queue_wrapper

        self._uvw_queue = create_queue_wrapper('uvw',
            QUEUE_SIZE, ['uvw', 'antenna1', 'antenna2'], ds)

        self._observation_queue = create_queue_wrapper('observation',
            QUEUE_SIZE, ['observed_vis', 'flag', 'weight'], ds)

        self._frequency_queue = create_queue_wrapper('frequency',
            QUEUE_SIZE, ['frequency', 'ref_frequency'], ds)

        self._die_queue = create_queue_wrapper('gterm',
            QUEUE_SIZE, ['gterm'], ds)

        self._dde_queue = create_queue_wrapper('dde',
            QUEUE_SIZE, ['ebeam', 'antenna_scaling', 'point_errors'], ds)

        self._point_source_queue = create_queue_wrapper('point_source',
            QUEUE_SIZE, ['point_lm', 'point_stokes', 'point_alpha'], ds)

        self._gaussian_source_queue = create_queue_wrapper('gaussian_source',
            QUEUE_SIZE, ['gaussian_lm', 'gaussian_stokes', 'gaussian_alpha',
                'gaussian_shape'], ds)

        self._sersic_source_queue = create_queue_wrapper('sersic_source',
            QUEUE_SIZE, ['sersic_lm', 'sersic_stokes', 'sersic_alpha',
                'sersic_shape'], ds)

        self._input_queue = create_queue_wrapper('input',
            QUEUE_SIZE, ['descriptor','model_vis'], ds)

        self._output_queue = create_queue_wrapper('output',
            QUEUE_SIZE, ['model_vis'], ds)

        self._parameter_queue = create_queue_wrapper('descriptors',
            QUEUE_SIZE, ['descriptor'], ds)

        #======================
        # Thread pool executors
        #======================
        self._parameter_executor = cf.ThreadPoolExecutor(1)
        self._feed_executor = cf.ThreadPoolExecutor(1)
        self._compute_executor = cf.ThreadPoolExecutor(1)

        #==========================
        # Tensorflow initialisation
        #==========================
        self._tf_expr = self._construct_tensorflow_expression()
        self._tf_session.run(tf.initialize_all_variables())

        #================
        # Cube Transcoder
        #================
        self._iter_dims = ['ntime', 'nbl']
        self._transcoder = CubeDimensionTranscoder(self._iter_dims)

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
        montblanc.log.info(("Selected a solver memory budget of {rb} "
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

    def _parameter_feed(self):
        try:
            self._parameter_feed_impl()
        except Exception as e:
            montblanc.log.exception("Parameter exception")
            raise

    def _parameter_feed_impl(self):
        session = self._tf_session

        # Copy dimensions of the main cube
        cube = HyperCube()
        cube.register_dimensions(self.dimensions(copy=False))

        # Iterate over time and baseline
        iter_strides = cube.dim_local_size(*self._iter_dims)
        iter_args = zip(self._iter_dims, iter_strides)

        # Iterate through the hypercube space
        for i, d in enumerate(cube.dim_iter(*iter_args, update_local_size=True)):
            cube.update_dimensions(d)
            descriptor = self._transcoder.encode(cube.dimensions(copy=False))
            feed_dict = {self._parameter_queue.placeholders[0] : descriptor }
            montblanc.log.debug('Encoding {i} {d}'.format(i=i, d=descriptor))
            session.run(self._parameter_queue.enqueue_op, feed_dict=feed_dict)

        # Close the queue
        session.run(self._parameter_queue.close())

    def _feed(self):
        """ Feed stub """
        try:
            self._feed_impl()
        except Exception as e:
            montblanc.log.exception("Feed exception")
            raise

    def _feed_impl(self):
        """ Implementation of queue feeding """
        session = self._tf_session
        data_sources = self._data_sources.copy()

        # Maintain a hypercube based on the main cube
        cube = HyperCube()
        cube.register_dimensions(self.dimensions(copy=False))
        cube.register_arrays(self.arrays())

        # Queues to be fed on each iteration
        chunk_queues = (self._input_queue,
            self._frequency_queue,
            self._uvw_queue,
            self._observation_queue,
            self._die_queue,
            self._dde_queue)

        chunks_read = 0

        src_queues  = {
            'npsrc' : self._point_source_queue,
            'ngsrc' : self._gaussian_source_queue,
            'nssrc' : self._sersic_source_queue,
        }

        while True:
            try:
                # Get the descriptor describing a portion of the RIME
                descriptor = session.run(self._parameter_queue.dequeue())

                # Decode the descriptor and update our cube dimensions
                dimensions = self._transcoder.decode(descriptor)
                cube.update_dimensions(dimensions)
                chunks_read += 1

            except tf.errors.OutOfRangeError as e:
                montblanc.log.info('Read {n} chunks'.format(n=chunks_read))
                break

            # Determine array shapes and data types for this
            # portion of the hypercube
            array_descriptors = cube.arrays(reify=True)

            # Inject data sources and array descriptors for the
            # descriptor queue items. These aren't full on arrays per se
            # but they need to work within the feeding framework
            array_descriptors['descriptor'] = descriptor
            data_sources['descriptor'] = DataSource(lambda c: descriptor, np.int32)

            # Generate (name, placeholder, datasource, array descriptor)
            # for the arrays required by each queue
            gen = [(a, ph, data_sources[a], array_descriptors[a])
                for q in chunk_queues
                for ph, a in zip(q.placeholders, q.fed_arrays)]

            # Create a feed dictionary by calling the data source functors
            feed_dict = { ph: ds.source(FeedContext(a, cube,
                    self.config(), ad.shape, ad.dtype))
                for (a, ph, ds, ad) in gen }

            montblanc.log.debug("Enqueueing chunk {i} {d}".format(
                i=chunks_read, d=descriptor))

            session.run([q.enqueue_op for q in chunk_queues],
                feed_dict=feed_dict)

            # For each source type, feed that source queue
            for src_type, queue in src_queues.iteritems():
                iter_args = (src_type, cube.dim_local_size(src_type))

                # Iterate over local_size chunks of the source 
                for dim_desc in cube.dim_iter(iter_args, update_local_size=True):
                    cube.update_dimensions(dim_desc)

                    montblanc.log.info("Enqueueing '{s}' '{t}' sources".format(
                        s=dim_desc[0]['local_size'], t=src_type))

                    # Generate (name, placeholder, datasource, array descriptor)
                    # for the arrays required by each queue
                    gen = [(a, ph, data_sources[a], array_descriptors[a])
                        for ph, a in zip(queue.placeholders, queue.fed_arrays)]

                    # Create a feed dictionary by calling the data source functors
                    feed_dict = { ph: ds.source(FeedContext(a, cube,
                            self.config(), ad.shape, ad.dtype))
                        for (a, ph, ds, ad) in gen }

                    session.run(queue.enqueue_op, feed_dict=feed_dict)

        montblanc.log.info('Done Feeding')

    def _compute_impl(self):
        """ Implementation of computation """
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        S = self._tf_session

        feed_dict = { ph: self.dim_global_size(n) for
            n, ph in self._src_ph_vars.iteritems() }

        feed_dict.update({ ph: getattr(self, n) for
            n, ph in self._property_ph_vars.iteritems() })

        result = S.run(self._tf_expr,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('compute-timeline.json', 'w') as f:
            f.write(ctf)

        return result

    def _compute(self):
        """ Compute stub """
        try:
            return self._compute_impl()
        except Exception as e:
            montblanc.log.exception("Compute exception")
            raise

    def _construct_tensorflow_expression(self):
        """ Constructs a tensorflow expression for computing the RIME """
        zero = tf.constant(0)

        # Pull RIME inputs out of the feed queues
        frequency, ref_frequency = self._frequency_queue.dequeue()
        descriptor, model_vis = self._input_queue.dequeue()
        uvw, antenna1, antenna2 = self._uvw_queue.dequeue()
        observed_vis, flag, weight = self._observation_queue.dequeue()
        ebeam, antenna_scaling, point_errors = self._dde_queue.dequeue()
        gterm = self._die_queue.dequeue()

        # Infer chunk dimensions
        model_vis_shape = tf.shape(model_vis)
        ntime, nbl, nchan = [model_vis_shape[i] for i in range(3)]

        def antenna_jones(lm, stokes, alpha):
            """
            Compute the jones terms for each antenna.

            lm, stokes and alpha are the source variables.
            """
            cplx_phase = rime.phase(lm, uvw, frequency, CT=model_vis.dtype)
            bsqrt = rime.b_sqrt(stokes, alpha, frequency, ref_frequency)
            ejones = rime.e_beam(lm, point_errors, antenna_scaling, ebeam,
                self._property_ph_vars.parallactic_angle,
                self._property_ph_vars.beam_ll,
                self._property_ph_vars.beam_lm,
                self._property_ph_vars.beam_ul,
                self._property_ph_vars.beam_um)

            return rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=lm.dtype)

        # While loop condition for each point source type
        def point_cond(model_vis, npsrc):
            return tf.less(npsrc, self._src_ph_vars.npsrc)

        def gaussian_cond(model_vis, ngsrc):
            return tf.less(ngsrc, self._src_ph_vars.ngsrc)

        def sersic_cond(model_vis, nssrc):
            return tf.less(nssrc, self._src_ph_vars.nssrc)

        # While loop bodies
        def point_body(model_vis, npsrc):
            """ Accumulate visiblities for point source batch """
            lm, stokes, alpha = self._point_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            ant_jones = antenna_jones(lm, stokes, alpha)
            shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=lm.dtype)    
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                shape, ant_jones, flag, gterm, model_vis, False)

            return model_vis, npsrc + nsrc

        def gaussian_body(model_vis, ngsrc):
            """ Accumulate visiblities for gaussian source batch """
            lm, stokes, alpha, gauss_params = self._gaussian_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            # Accumulate visiblities for this source batch

            ant_jones = antenna_jones(lm, stokes, alpha)
            gauss_shape = rime.gauss_shape(uvw, antenna1, antenna1,
                frequency, gauss_params)
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                gauss_shape, ant_jones, flag, gterm, model_vis, False)

            return model_vis, ngsrc + nsrc

        def sersic_body(model_vis, nssrc):
            """ Accumulate visiblities for sersic source batch """
            lm, stokes, alpha, sersic_params = self._sersic_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            # Accumulate visiblities for this source batch
            ant_jones = antenna_jones(lm, stokes, alpha)
            sersic_shape = rime.sersic_shape(uvw, antenna1, antenna1,
                frequency, sersic_params)
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                sersic_shape, ant_jones, flag, gterm, model_vis, False)

            return model_vis, nssrc + nsrc

        # Evaluate point sources
        model_vis, npsrc = tf.while_loop(point_cond, point_body,
            [model_vis, zero])

        # Evaluate gaussians
        model_vis, ngsrc = tf.while_loop(gaussian_cond, gaussian_body,
            [model_vis, zero])

        # Evaluate sersics
        model_vis, nssrc = tf.while_loop(sersic_cond, sersic_body,
            [model_vis, zero])

        return descriptor, model_vis

    def solve(self):
        p = self._parameter_executor.submit(self._parameter_feed)
        f = self._feed_executor.submit(self._feed)
        c = self._compute_executor.submit(self._compute)

        done, not_done = cf.wait([f,c], return_when=cf.FIRST_COMPLETED)

        for d in done:
            data = d.result()
            print data
            print data.shape

    def close(self):
        self._parameter_executor.shutdown()
        self._feed_executor.shutdown()
        self._compute_executor.shutdown()
        self._tf_session.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()