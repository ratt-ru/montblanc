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
from montblanc.impl.rime.tensorflow.cube_dim_transcoder import CubeDimensionTranscoder

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager

from montblanc.impl.rime.tensorflow.sources import (SourceContext,
                                                    MSSourceProvider, FitsBeamSourceProvider)

# TODO: Move this into a separate package
from montblanc.impl.rime.tensorflow.sources.queue_wrapper import (
    create_queue_wrapper)

from montblanc.impl.rime.tensorflow.sinks import (SinkContext,
                                                  NullSinkProvider, MSSinkProvider)

from hypercube import HyperCube
import hypercube.util as hcu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

rime_lib_path = os.path.join(montblanc.get_montblanc_path(),
    'tensorflow', 'rime_ops', 'rime.so')
rime = tf.load_op_library(rime_lib_path)

DataSource = collections.namedtuple("DataSource", ['source', 'dtype', 'name'])
DataSink = collections.namedtuple("DataSink", ['sink', 'name'])

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

        #=========================================
        # Register hypercube Arrays and Properties
        #=========================================

        from montblanc.impl.rime.tensorflow.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        #===================
        # Tensorflow Session
        #===================

        # Create the tensorflow session object
        self._tf_session = tf.Session()

        #================================
        # Queue Data Source Configuration
        #================================

        # Get the data source (defaults or test data)
        data_source = slvr_cfg.get(Options.DATA_SOURCE)

        # Set up the queue data sources. Just take from
        # defaults if test data isn't specified
        queue_data_source = (Options.DATA_SOURCE_DEFAULT
            if not data_source == Options.DATA_SOURCE_TEST
            else data_source)

        # Obtain default data sources for each array,
        # then update with any data sources supplied by the user

        self._default_data_sources = dfs = {
            n: DataSource(a.get(queue_data_source), a.dtype, queue_data_source)
            for n, a in self.arrays().iteritems()
            if not a.temporary }

        montblanc.log.info("Data source '{dfs}'".format(dfs=data_source))

        # The descriptor queue items are not user-defined arrays
        # but a variable passed through describing a chunk of the
        # problem. Make it look as if it's an array
        if 'descriptor' in dfs:
            raise KeyError("'descriptor' is reserved, "
                "please use another array name.")

        dfs['descriptor'] = DataSource(lambda c: np.int32([0]), np.int32, 'Internal')

        #==================
        # Data Sources
        #==================

        self._ms_manager = None

        # Construct list of data sources internal to the solver
        # Any data sources specified in the solve() method will
        # override these
        self._source_providers = []

        # Create and add the FITS Beam Data Source, if present
        fits_filename_schema = slvr_cfg.get(Options.E_BEAM_FITS_FILENAMES, '')

        if fits_filename_schema:
            self._source_providers.append(FitsBeamSourceProvider(
                fits_filename_schema))

        # Create and add the MS Data Source
        if data_source == Options.DATA_SOURCE_MS:
            msfile = slvr_cfg.get(Options.MS_FILE)
            self._ms_manager = mgr = MeasurementSetManager(msfile, self, slvr_cfg)
            self._source_providers.append(MSSourceProvider(mgr))

        #==================
        # Data Sinks
        #==================

        # Construct list of data sinks internal to the solver
        # Any data sinks specified in the solve() method will
        # override these.
        self._sink_providers = [NullSinkProvider()]

        # We have an MS so add a MS data sink
        if self._ms_manager is not None:
            self._sink_providers.append(MSSinkProvider(self._ms_manager))

        #==================
        # Memory Budgeting
        #==================

        # For deciding whether to rebudget
        self._previous_budget = 0

        #======================
        # Tensorflow Placeholders
        #======================

        # Create placholder variables for source counts
        self._src_ph_vars = AttrDict({
            n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
            for n in ['nsrc'] + mbu.source_nr_vars()})

        # Create placeholder variables for properties
        self._property_ph_vars = AttrDict({
            n: tf.placeholder(dtype=p.dtype, shape=(), name=n)
            for n, p in self.properties().iteritems() })

        #======================
        # Thread pool executors
        #======================
        self._parameter_executor = cf.ThreadPoolExecutor(1)
        self._feed_executor = cf.ThreadPoolExecutor(1)
        self._compute_executor = cf.ThreadPoolExecutor(1)
        self._consumer_executor = cf.ThreadPoolExecutor(1)

        #==========================
        # Tensorflow initialisation
        #==========================
        self._tf_expr = self._construct_tensorflow_expression(dfs)
        self._tf_session.run(tf.initialize_all_variables())

        #================
        # Cube Transcoder
        #================
        self._iter_dims = ['ntime', 'nbl']
        self._transcoder = CubeDimensionTranscoder(self._iter_dims)

    def _parameter_feed(self):
        try:
            self._parameter_feed_impl()
        except Exception as e:
            montblanc.log.exception("Parameter Exception")
            raise

    def _parameter_feed_impl(self):
        session = self._tf_session

        # Copy dimensions of the main cube
        cube = self.copy()

        # Get space of iteration
        iter_args = _iter_args(self._iter_dims, cube)
        parameters_fed = 0

        # Iterate through the hypercube space
        for i, d in enumerate(cube.dim_iter(*iter_args, update_local_size=True)):
            cube.update_dimensions(d)
            descriptor = self._transcoder.encode(cube.dimensions(copy=False))
            feed_dict = {self._parameter_queue.placeholders[0] : descriptor }
            montblanc.log.debug('Encoding {i} {d}'.format(i=i, d=descriptor))
            session.run(self._parameter_queue.enqueue_op, feed_dict=feed_dict)
            parameters_fed += 1

        montblanc.log.info("Done feeding {n} parameters.".format(
            n=parameters_fed))

    def _feed(self, source_providers):
        """ Feed stub """
        try:
            self._feed_impl(source_providers)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_impl(self, source_providers):
        """ Implementation of queue feeding """
        session = self._tf_session

        # Maintain a hypercube based on the main cube
        cube = self.copy()

        # Get space of iteration
        global_iter_args = _iter_args(self._iter_dims, cube)

        # Construct data sources from those supplied by the
        # source providers, if they're associated with
        # input queue arrays
        data_sources = self._default_data_sources.copy()
        data_sources.update({
            n: DataSource(f, cube.array(n).dtype, source.name())
            for source in source_providers
            for n, f in source.sources().iteritems()
            if n in self._queue_arrays})


        chunks_fed = 0
        done = False

        while not done:
            try:
                # Get the descriptor describing a portion of the RIME
                descriptor = session.run(self._parameter_queue.dequeue_op)
            except tf.errors.OutOfRangeError as e:
                montblanc.log.exception("Descriptor reading exception")

            # Decode the descriptor and update our cube dimensions
            dims = self._transcoder.decode(descriptor)
            # Are we done?
            done = _last_chunk(dims)
            cube.update_dimensions(dims)

            # Determine array shapes and data types for this
            # portion of the hypercube
            array_schemas = cube.arrays(reify=True)

            # Inject a data source and array schema for the
            # descriptor queue items. These aren't full on arrays per se
            # but they need to work within the feeding framework
            array_schemas['descriptor'] = descriptor
            data_sources['descriptor'] = DataSource(lambda c: descriptor, np.int32, 'Internal')

            def _get_data(data_source, context):
                # Invoke the data source
                data = data_source.source(context)

                # Complain about None values
                if data is None:
                    raise ValueError("'None' returned from "
                        "data source '{n}'".format(n=context.name))

                # Check that the data matches the expected shape
                same = (data.shape == context.shape and
                        data.dtype == context.dtype)

                if not same:
                    raise ValueError("Expected data of shape '{esh}' and "
                        "dtype '{edt}' for data source '{n}', but "
                        "shape '{rsh}' and '{rdt}' was found instead".format(
                            n=context.name,
                            esh=context.shape, edt=context.dtype,
                            rsh=data.shape, rdt=data.dtype))

                return data

            # Generate (name, placeholder, datasource, array schema)
            # for the arrays required by each queue
            gen = [(a, ph, data_sources[a], array_schemas[a])
                for q in self._chunk_queues
                for ph, a in zip(q.placeholders, q.fed_arrays)]

            # Create a feed dictionary by calling the data source functors
            feed_dict = { ph: _get_data(ds, SourceContext(a, cube,
                    self.config(), global_iter_args,
                    cube.array(a) if a in cube.arrays() else {},
                    ad.shape, ad.dtype))
                for (a, ph, ds, ad) in gen }

            montblanc.log.debug("Enqueueing chunk {i} {d}".format(
                i=chunks_fed, d=descriptor))

            session.run([q.enqueue_op for q in self._chunk_queues],
                feed_dict=feed_dict)

            chunks_fed += 1

            # For each source type, feed that source queue
            for src_type, queue in self._src_queues.iteritems():
                iter_args = [(src_type, cube.dim_local_size(src_type))]

                # Iterate over local_size chunks of the source
                for dim_desc in cube.dim_iter(*iter_args, update_local_size=True):
                    cube.update_dimensions(dim_desc)

                    montblanc.log.debug("Enqueueing '{s}' '{t}' sources".format(
                        s=dim_desc[0]['local_size'], t=src_type))

                    # Generate (name, placeholder, datasource, array descriptor)
                    # for the arrays required by each queue
                    gen = [(a, ph, data_sources[a], array_schemas[a])
                        for ph, a in zip(queue.placeholders, queue.fed_arrays)]

                    # Create a feed dictionary by calling the data source functors
                    feed_dict = { ph: _get_data(ds, SourceContext(a, cube,
                            self.config(), global_iter_args + iter_args,
                            cube.array(a) if a in cube.arrays() else {},
                            ad.shape, ad.dtype))
                        for (a, ph, ds, ad) in gen }

                    session.run(queue.enqueue_op, feed_dict=feed_dict)

        montblanc.log.info("Done feeding {n} chunks.".format(n=chunks_fed))

    def _compute_impl(self):
        """ Implementation of computation """
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
        #    timeout_in_ms=10000)
        run_options =tf.RunOptions()
        run_metadata = tf.RunMetadata()

        S = self._tf_session
        chunks_computed = 0
        done = False

        feed_dict = { ph: self.dim_global_size(n) for
            n, ph in self._src_ph_vars.iteritems() }

        feed_dict.update({ ph: getattr(self, n) for
            n, ph in self._property_ph_vars.iteritems() })

        while not done:
            descriptor, enq = S.run(self._tf_expr,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)

            # Are we done?
            dims = self._transcoder.decode(descriptor)
            done = _last_chunk(dims)
            chunks_computed += 1

        montblanc.log.info("Done computing {n} chunks."
            .format(n=chunks_computed))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('compute-timeline.json', 'w') as f:
            f.write(ctf)

    def _compute(self):
        """ Compute stub """
        try:
            return self._compute_impl()
        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise

    def _consume(self, sink_providers):
        """ Consume stub """
        try:
            return self._consume_impl(sink_providers)
        except Exception as e:
            montblanc.log.exception("Consumer Exception")
            raise

    def _consume_impl(self, sink_providers):
        """ Consume """

        S = self._tf_session
        chunks_consumed = 0
        done = False

        # Maintain a hypercube based on the main cube
        cube = self.copy()

        # Get space of iteration
        global_iter_args = _iter_args(self._iter_dims, cube)

        # Get data sinks from supplied providers
        data_sinks = { n: DataSink(f, sink.name())
            for sink in sink_providers
            for n, f in sink.sinks().iteritems()
            if not n == 'descriptor' }


        while not done:
            output = S.run(self._output_queue.dequeue_op)
            chunks_consumed += 1

            # Expect the descriptor in the first tuple position
            assert len(output) > 0
            assert self._output_queue.fed_arrays[0] == 'descriptor'

            descriptor = output[0]
            dims = self._transcoder.decode(descriptor)
            cube.update_dimensions(dims)

            # For each array in our output, call the associated data sink
            for n, a in zip(self._output_queue.fed_arrays[1:], output[1:]):
                sink_context = SinkContext(n, cube, self.config(), global_iter_args,
                    cube.array(n) if n in cube.arrays() else {}, a)
                data_sinks[n].sink(sink_context)

            # Are we done?
            done = _last_chunk(dims)

        montblanc.log.info('Done consuming {n} chunks'.format(n=chunks_consumed))

    def _construct_tensorflow_expression(self, dfs):
        """ Constructs a tensorflow expression for computing the RIME """

        QUEUE_SIZE = 10

        # TODO: don't create these queues on the object instance,
        # instead we should return them
        self._parameter_queue = create_queue_wrapper('descriptors',
            QUEUE_SIZE, ['descriptor'], dfs)

        self._input_queue = create_queue_wrapper('input',
            QUEUE_SIZE, ['descriptor','model_vis'], dfs)

        self._uvw_queue = create_queue_wrapper('uvw',
            QUEUE_SIZE, ['uvw', 'antenna1', 'antenna2'], dfs)

        self._observation_queue = create_queue_wrapper('observation',
            QUEUE_SIZE, ['observed_vis', 'flag', 'weight'], dfs)

        self._frequency_queue = create_queue_wrapper('frequency',
            QUEUE_SIZE, ['frequency', 'ref_frequency'], dfs)

        self._die_queue = create_queue_wrapper('gterm',
            QUEUE_SIZE, ['gterm'], dfs)

        self._dde_queue = create_queue_wrapper('dde',
            QUEUE_SIZE, ['ebeam', 'antenna_scaling', 'point_errors',
                'parallactic_angles', 'beam_extents'], dfs)

        self._point_source_queue = create_queue_wrapper('point_source',
            QUEUE_SIZE, ['point_lm', 'point_stokes', 'point_alpha'], dfs)

        self._gaussian_source_queue = create_queue_wrapper('gaussian_source',
            QUEUE_SIZE, ['gaussian_lm', 'gaussian_stokes', 'gaussian_alpha',
                'gaussian_shape'], dfs)

        self._sersic_source_queue = create_queue_wrapper('sersic_source',
            QUEUE_SIZE, ['sersic_lm', 'sersic_stokes', 'sersic_alpha',
                'sersic_shape'], dfs)

        self._output_queue = create_queue_wrapper('output',
            QUEUE_SIZE, ['descriptor', 'model_vis'], dfs)

        # TODO: don't create objects on the object instance,
        # instead we should return them

        # Source queues to feed
        self._src_queues  = {
            'npsrc' : self._point_source_queue,
            'ngsrc' : self._gaussian_source_queue,
            'nssrc' : self._sersic_source_queue,
        }

        # Visibility chunk queues to feed
        self._chunk_queues = [self._input_queue,
            self._frequency_queue,
            self._uvw_queue,
            self._observation_queue,
            self._die_queue,
            self._dde_queue]

        # Set of arrays that the queues feed
        self._queue_arrays = { a
            for q in self._chunk_queues + self._src_queues.values()
            for a in q.fed_arrays }

        zero = tf.constant(0)
        src_count = zero

        # Pull RIME inputs out of the feed queues
        frequency, ref_frequency = self._frequency_queue.dequeue()
        descriptor, model_vis = self._input_queue.dequeue()
        uvw, antenna1, antenna2 = self._uvw_queue.dequeue()
        observed_vis, flag, weight = self._observation_queue.dequeue()
        gterm = self._die_queue.dequeue()
        (ebeam, antenna_scaling, point_errors,
            parallactic_angles, beam_extents) = self._dde_queue.dequeue()

        # Infer chunk dimensions
        model_vis_shape = tf.shape(model_vis)
        ntime, nbl, nchan = [model_vis_shape[i] for i in range(3)]
        FT, CT = uvw.dtype, model_vis.dtype

        def apply_dies(src_count):
            """ Have we reached the maximum source count """
            return tf.greater_equal(src_count, self._src_ph_vars.nsrc)

        def antenna_jones(lm, stokes, alpha):
            """
            Compute the jones terms for each antenna.

            lm, stokes and alpha are the source variables.
            """
            cplx_phase = rime.phase(lm, uvw, frequency, CT=CT)
            bsqrt = rime.b_sqrt(stokes, alpha, frequency, ref_frequency, CT=CT)
            ejones = rime.e_beam(lm, frequency,
                point_errors, antenna_scaling,
                parallactic_angles,
                beam_extents, ebeam)

            return rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=FT)

        # While loop condition for each point source type
        def point_cond(model_vis, npsrc, src_count):
            return tf.less(npsrc, self._src_ph_vars.npsrc)

        def gaussian_cond(model_vis, ngsrc, src_count):
            return tf.less(ngsrc, self._src_ph_vars.ngsrc)

        def sersic_cond(model_vis, nssrc, src_count):
            return tf.less(nssrc, self._src_ph_vars.nssrc)

        # While loop bodies
        def point_body(model_vis, npsrc, src_count):
            """ Accumulate visiblities for point source batch """
            lm, stokes, alpha = self._point_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            src_count += nsrc
            npsrc +=  nsrc
            ant_jones = antenna_jones(lm, stokes, alpha)
            shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=FT)
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                shape, ant_jones, flag, gterm, model_vis,
                apply_dies(src_count))

            return model_vis, npsrc, src_count

        def gaussian_body(model_vis, ngsrc, src_count):
            """ Accumulate visiblities for gaussian source batch """
            lm, stokes, alpha, gauss_params = self._gaussian_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            src_count += nsrc
            ngsrc += nsrc
            ant_jones = antenna_jones(lm, stokes, alpha)
            gauss_shape = rime.gauss_shape(uvw, antenna1, antenna1,
                frequency, gauss_params)
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                gauss_shape, ant_jones, flag, gterm, model_vis,
                apply_dies(src_count))

            return model_vis, ngsrc, src_count

        def sersic_body(model_vis, nssrc, src_count):
            """ Accumulate visiblities for sersic source batch """
            lm, stokes, alpha, sersic_params = self._sersic_source_queue.dequeue()
            nsrc = tf.shape(lm)[0]
            src_count += nsrc
            nssrc += nsrc
            ant_jones = antenna_jones(lm, stokes, alpha)
            sersic_shape = rime.sersic_shape(uvw, antenna1, antenna1,
                frequency, sersic_params)
            model_vis = rime.sum_coherencies(antenna1, antenna2,
                sersic_shape, ant_jones, flag, gterm, model_vis,
                apply_dies(src_count))

            return model_vis, nssrc, src_count

        # Evaluate point sources
        model_vis, npsrc, src_count = tf.while_loop(
            point_cond, point_body,
            [model_vis, zero, src_count])

        # Evaluate gaussians
        model_vis, ngsrc, src_count = tf.while_loop(
            gaussian_cond, gaussian_body,
            [model_vis, zero, src_count])

        # Evaluate sersics
        model_vis, nssrc, src_count = tf.while_loop(
            sersic_cond, sersic_body,
            [model_vis, zero, src_count])

        # Create enqueue operation
        enqueue_op = self._output_queue.queue.enqueue([descriptor, model_vis])

        # Return descriptor and enqueue operation
        return descriptor, enqueue_op

    def solve(self, *args, **kwargs):
        #  Obtain source and sink providers, including internal providers
        source_providers = (self._source_providers +
            kwargs.get('source_providers', []))
        sink_providers = (self._sink_providers +
            kwargs.get('sink_providers', []))

        print 'Source Providers', source_providers
        print 'Sink Providers', sink_providers

        # Apply any dimension updates from the source provider
        # to the hypercube
        bytes_required = _apply_source_provider_dim_updates(self,
            source_providers)

        # If we use more memory than previously,
        # perform another budgeting operation
        # to make sure everything fits
        if bytes_required > self._previous_budget:
            self._previous_budget = _budget(self, self.config())

        try:
            params = self._parameter_executor.submit(self._parameter_feed)
            feed = self._feed_executor.submit(self._feed, source_providers)
            compute = self._compute_executor.submit(self._compute)
            consume = self._consumer_executor.submit(self._consume, sink_providers)

            not_done = [params, feed, compute, consume]

            while True:
                # TODO: timeout not strictly necessary
                done, not_done = cf.wait(not_done,
                    timeout=1.0,
                    return_when=cf.FIRST_COMPLETED)

                for future in done:
                    res = future.result()

                # Nothing remains to be done, quit the loop
                if len(not_done) == 0:
                    break

        except (KeyboardInterrupt, SystemExit) as e:
            montblanc.log.exception('Solving interrupted')
            raise
        except:
            montblanc.log.exception('Solving exception')

    def close(self):
        # Shutdown thread executors
        self._parameter_executor.shutdown()
        self._feed_executor.shutdown()
        self._compute_executor.shutdown()
        self._consumer_executor.shutdown()

        # Shutdown thte tensorflow session
        self._tf_session.close()

        # Shutdown data sources
        for source in self._source_providers:
            source.close()

        # Shutdown data sinks
        for sink in self._sink_providers:
            sink.close()

        # Close the measurement set manager
        if self._ms_manager is not None:
            self._ms_manager.close()


    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()


def _last_chunk(dims):
    """
    Does the list of dimension dictionaries indicate this is the last chunk?
    i.e. does upper_extent == global_size for all dimensions?
    """
    return all(d['upper_extent'] == d['global_size'] for d in dims)

def _iter_args(iter_dims, cube):
    iter_strides = cube.dim_local_size(*iter_dims)
    return zip(iter_dims, iter_strides)

def _uniq_log2_range(start, size, div):
    start = np.log2(start)
    size = np.log2(size)
    int_values = np.int32(np.logspace(start, size, div, base=2)[:-1])

    return np.flipud(np.unique(int_values))

BUDGETING_DIMS = ['nbl', 'ntime', 'nsrc'] + mbu.source_nr_vars()

def _budget(cube, slvr_cfg):
    # Figure out a viable dimension configuration
    # given the total problem size
    mem_budget = slvr_cfg.get('mem_budget', 2*ONE_GB)
    bytes_required = cube.bytes_required()

    dim_names = 'na', 'nbl', 'ntime', 'nsrc'
    global_sizes =  na, nbl, ntime, nsrc = cube.dim_global_size(*dim_names)

    # Keep track of original dimension sizes and any reductions that are applied
    # Ignore 'na'
    original_sizes = { r: s for r, s in zip(dim_names, global_sizes)[1:] }
    applied_reductions = {}

    def _reduction():
        # Reduce over time first
        for t in _uniq_log2_range(1, ntime, 5):
            yield [('ntime', t)]

        # Attempt reduction over source
        bs = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)

        if nsrc > bs:
            yield [(d,nsrc) for d in [Options.NSRC] + mbu.source_nr_vars()]

        # Reduce by baseline
        for bl in _uniq_log2_range(na, nbl, 5):
            yield [('nbl', bl)]

    for reduction in _reduction():
        if bytes_required > mem_budget:
            for dim, size in reduction:
                applied_reductions[dim] = size
                cube.update_dimension(dim, local_size=size,
                    lower_extent=0, upper_extent=size)
        else:
            break

        bytes_required = cube.bytes_required()

    # Log some information about the memory_budget
    # and dimension reduction
    montblanc.log.info(("Selected a solver memory budget of {rb} "
        "given a hard limit of {mb}.").format(
        rb=mbu.fmt_bytes(bytes_required),
        mb=mbu.fmt_bytes(mem_budget)))

    if len(applied_reductions) > 0:
        montblanc.log.info("The following dimension reductions "
            "were applied:")

        for k, v in applied_reductions.iteritems():
            montblanc.log.info('{p}{d}: {id} => {rd}'.format
                (p=' '*4, d=k, id=original_sizes[k], rd=v))
    else:
        montblanc.log.info("No dimension reductions were applied.")

    return bytes_required

def _apply_source_provider_dim_updates(cube, source_providers):
    """
    Given a list of source_providers, apply the list of
    suggested dimension updates given in provider.updated_dimensions()
    to the supplied hypercube.

    Dimension global sizes are always updated. Local sizes will be applied
    UNLESS the dimension is reduced during the budgeting process.
    Assumption here is that the budgeter is smarter than the user.

    """
    # Create a mapping between a dimension and a
    # list of (global_size, provider_name) tuples
    mapping = collections.defaultdict(list)

    # Update the mapping, except for the nsrc dimension
    [mapping[d.name].append((d, prov.name()))
        for prov in source_providers
        for d in prov.updated_dimensions()
        if not d.name == 'nsrc' ]

    # Ensure that the global sizes we receive
    # for each dimension are unique. Tell the user
    # about which sources conflict
    for n, u in mapping.iteritems():
        if not all(u[0][0] == tup[0] for tup in u[1:]):
            raise ValueError("Received conflicting global size updates '{u}'"
                " for dimension '{n}'.".format(n=n, u=u))

    if len(mapping) > 0:
        montblanc.log.info("Updating dimensions {mk} from "
            "source providers.".format(mk=mapping.keys()))

    # Get existing local dimension sizes
    local_sizes = cube.dim_local_size(*mapping.keys())

    # Now update our dimensions
    for (n, u), ls in zip(mapping.iteritems(), local_sizes):
        # Reduce our local size to satisfy hypercube
        d = u[0][0]
        gs = d.global_size
        # Defer to existing local size for budgeting dimensions
        ls = ls if ls in BUDGETING_DIMS else d.local_size
        # Clamp local size to global size
        ls = gs if ls > gs else ls
        cube.update_dimension(n, local_size=ls, global_size=gs,
            lower_extent=0, upper_extent=ls)

    # Handle total number of sources differently
    # It's equal to the number of
    # point's, gaussian's, sersic's combined
    nsrc = sum(cube.dim_global_size(*mbu.source_nr_vars()))
    ls = cube.dim_local_size('nsrc')
    ls = nsrc if ls > nsrc else ls

    cube.update_dimension('nsrc',
        local_size=ls, global_size=nsrc,
        lower_extent=0, upper_extent=ls)

    # Return our cube size
    return cube.bytes_required()
