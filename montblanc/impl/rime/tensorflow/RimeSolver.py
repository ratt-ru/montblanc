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

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager

from montblanc.impl.rime.tensorflow.sources import (SourceContext,
    MSRimeDataSource, FitsBeamDataSource)

# TODO: Move this into a separate package
from montblanc.impl.rime.tensorflow.sources.queue_wrapper import (
    create_queue_wrapper)

from montblanc.impl.rime.tensorflow.sinks import (SinkContext,
    NullDataSink, MSRimeDataSink)

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

        # Monkey patch these functions onto the object
        monkey_patch_antenna_pairs(self)

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

        QUEUE_SIZE = 10

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

        #==================
        # Data Sources
        #==================

        self._ms_manager = None

        # Construct list of data sources internal to the solver
        # Any data sources specified in the solve() method will
        # override these
        self._sources = []

        # Create and add the FITS Beam Data Source, if present
        base_fits_name = slvr_cfg.get(Options.E_BEAM_BASE_FITS_NAME, '')

        if base_fits_name:
            self._source.append(FitsBeamDataSource(base_fits_name))

        # Create and add the MS Data Source
        if data_source == Options.DATA_SOURCE_MS:
            msfile = slvr_cfg.get(Options.MS_FILE)
            self._ms_manager = mgr = MeasurementSetManager(msfile, self)
            self._sources.append(MSRimeDataSource(mgr))

        # Use any dimension update hints from the sources
        for data_source in self._sources:
            self.update_dimensions(data_source.updated_dimensions())

        #==================
        # Data Sinks
        #==================

        # Construct list of data sinks internal to the solver
        # Any data sinks specified in the solve() method will
        # override these.
        self._sinks = [NullDataSink()]

        # We have an MS so add a MS data sink
        if self._ms_manager is not None:
            self._sinks.append(MSRimeDataSink(self._ms_manager))

        #==================
        # Memory Budgeting
        #==================

        # Attempt to fit arrays into memory budget by
        # reducing dimension local_sizes
        self._modded_dims = modded_dims = self._budget(A, slvr_cfg)

        # Update any dimensions
        for k, v in modded_dims.iteritems():
            self.update_dimension(k, local_size=v,
                lower_extent=0, upper_extent=v)

        #======================
        # Tensorflow Placeholders
        #======================

        # Create placholder variables for source counts
        self._src_ph_vars = AttrDict({
            n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
            for n in mbu.source_nr_vars() })

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
        self._tf_expr = self._construct_tensorflow_expression()
        self._tf_session.run(tf.initialize_all_variables())

        #================
        # Cube Transcoder
        #================
        self._iter_dims = ['ntime', 'nbl']
        self._transcoder = CubeDimensionTranscoder(self._iter_dims)

    def _budget(self, arrays, slvr_cfg):
        nsrc = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)
        src_str_list = [Options.NSRC] + mbu.source_nr_vars()
        src_reduction_str = '&'.join(['%s=%s' % (nr_var, nsrc)
            for nr_var in src_str_list])

        mem__budget = slvr_cfg.get('mem_budget', 256*ONE_MB)
        T = self.template_dict()
        na = self.dim_local_size('na')

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

        if len(modded_dims) > 0:
            montblanc.log.info((
                "The following dimension reductions "
                "have been applied:"))

            for k, v in modded_dims.iteritems():
                montblanc.log.info('{p}{d}: {id} => {rd}'.format
                    (p=' '*4, d=k, id=T[k], rd=v))
        else:
            montblanc.log.info("No dimension reductions were applied.")

        return modded_dims

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

        # Iterate over time and baseline
        iter_strides = cube.dim_local_size(*self._iter_dims)
        iter_args = zip(self._iter_dims, iter_strides)
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

    def _feed(self, data_sources):
        """ Feed stub """
        try:
            self._feed_impl(data_sources)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_impl(self, data_sources):
        """ Implementation of queue feeding """
        session = self._tf_session

        # Maintain a hypercube based on the main cube
        cube = self.copy()

        # Construct data sources
        data_sources = self._sources + data_sources

        # Construct per array data sources
        _data_sources = self._default_data_sources.copy()
        _data_sources.update({
            n: DataSource(f, cube.array(n).dtype, source.name())
            for source in data_sources
            for n, f in source.sources().iteritems()})

        data_sources = _data_sources

        # Queues to be fed on each iteration
        chunk_queues = (self._input_queue,
            self._frequency_queue,
            self._uvw_queue,
            self._observation_queue,
            self._die_queue,
            self._dde_queue)

        chunks_fed = 0
        done = False

        src_queues  = {
            'npsrc' : self._point_source_queue,
            'ngsrc' : self._gaussian_source_queue,
            'nssrc' : self._sersic_source_queue,
        }

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

            # Generate (name, placeholder, datasource, array descriptor)
            # for the arrays required by each queue
            gen = [(a, ph, data_sources[a], array_schemas[a])
                for q in chunk_queues
                for ph, a in zip(q.placeholders, q.fed_arrays)]

            def _get_data_source(data_source, context):
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

            # Create a feed dictionary by calling the data source functors
            feed_dict = { ph: _get_data_source(ds,
                    SourceContext(a, cube, self.config(), ad.shape, ad.dtype))
                for (a, ph, ds, ad) in gen }

            montblanc.log.debug("Enqueueing chunk {i} {d}".format(
                i=chunks_fed, d=descriptor))

            session.run([q.enqueue_op for q in chunk_queues],
                feed_dict=feed_dict)

            chunks_fed += 1

            # For each source type, feed that source queue
            for src_type, queue in src_queues.iteritems():
                iter_args = (src_type, cube.dim_local_size(src_type))

                # Iterate over local_size chunks of the source
                for dim_desc in cube.dim_iter(iter_args, update_local_size=True):
                    cube.update_dimensions(dim_desc)

                    montblanc.log.debug("Enqueueing '{s}' '{t}' sources".format(
                        s=dim_desc[0]['local_size'], t=src_type))

                    # Generate (name, placeholder, datasource, array descriptor)
                    # for the arrays required by each queue
                    gen = [(a, ph, data_sources[a], array_schemas[a])
                        for ph, a in zip(queue.placeholders, queue.fed_arrays)]

                    # Create a feed dictionary by calling the data source functors
                    feed_dict = { ph: ds.source(SourceContext(a, cube,
                            self.config(), ad.shape, ad.dtype))
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

    def _consume(self, data_sinks):
        """ Consume stub """
        try:
            return self._consume_impl(data_sinks)
        except Exception as e:
            montblanc.log.exception("Consumer Exception")
            raise

    def _consume_impl(self, data_sinks):
        """ Consume """

        S = self._tf_session
        chunks_consumed = 0
        done = False

        # Maintain a hypercube based on the main cube
        cube = self.copy()

        # Construct our data sinks
        data_sinks = self._sinks + data_sinks

        # Construct per array data sinks
        _data_sinks = { n: DataSink(f, sink.name())
            for sink in data_sinks
            for n, f in sink.sinks().iteritems()
            if not n == 'descriptor' }

        data_sinks = _data_sinks

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
                sink_context = SinkContext(n, cube, self.config(), a)
                data_sinks[n].sink(sink_context)

            # Are we done?
            done = _last_chunk(dims)

        montblanc.log.info('Done consuming {n} chunks'.format(n=chunks_consumed))

    def _construct_tensorflow_expression(self):
        """ Constructs a tensorflow expression for computing the RIME """
        zero = tf.constant(0)

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

        def antenna_jones(lm, stokes, alpha):
            """
            Compute the jones terms for each antenna.

            lm, stokes and alpha are the source variables.
            """
            cplx_phase = rime.phase(lm, uvw, frequency,
                CT=model_vis.dtype)
            bsqrt = rime.b_sqrt(stokes, alpha, frequency, ref_frequency,
                CT=model_vis.dtype)
            ejones = rime.e_beam(lm, frequency,
                point_errors, antenna_scaling,
                parallactic_angles, beam_extents, ebeam)

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

        # Create enqueue operation
        enqueue_op = self._output_queue.queue.enqueue([descriptor, model_vis])

        # Return descriptor and enqueue operation
        return descriptor, enqueue_op

    def solve(self, *args, **kwargs):

        data_sources = kwargs.get('data_sources', [])
        data_sinks = kwargs.get('data_sinks', [])

        try:
            params = self._parameter_executor.submit(self._parameter_feed)
            feed = self._feed_executor.submit(self._feed, data_sources)
            compute = self._compute_executor.submit(self._compute)
            consume = self._consumer_executor.submit(self._consume, data_sinks)

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
        for source in self._sources:
            source.close()

        # Shutdown data sinks
        for sink in self._sinks:
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
