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
import copy
import itertools
import threading
import sys

import concurrent.futures as cf
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from attrdict import AttrDict
import attr

import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow import load_tf_lib
from montblanc.impl.rime.tensorflow.cube_dim_transcoder import CubeDimensionTranscoder
from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
from montblanc.impl.rime.tensorflow.sources import (SourceContext,
                                                    MSSourceProvider, FitsBeamSourceProvider)
from montblanc.impl.rime.tensorflow.queue_wrapper import (
    create_queue_wrapper)

from montblanc.impl.rime.tensorflow.sinks import (SinkContext,
                                                  NullSinkProvider, MSSinkProvider)

from hypercube.dims import Dimension as HyperCubeDim

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

rime = load_tf_lib()

DataSource = attr.make_class("DataSource", ['source', 'dtype', 'name'],
    slots=True, frozen=True)
DataSink = attr.make_class("DataSink", ['sink', 'name'],
    slots=True, frozen=True)
FeedOnce = attr.make_class("FeedOnce", ['ph', 'var', 'assign_op'],
    slots=True, frozen=True)

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

        cube, slvr_cfg = self.hypercube, self.config()

        mbu.register_default_dimensions(cube, slvr_cfg)

        # Configure the dimensions of the beam cube
        cube.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube l width')

        cube.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube m height')

        cube.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube nu depth')

        #=========================================
        # Register hypercube Arrays and Properties
        #=========================================

        from montblanc.impl.rime.tensorflow.config import (A, P)

        def _massage_dtypes(A, T):
            def _massage_dtype_in_dict(D):
                new_dict = D.copy()
                new_dict['dtype'] = mbu.dtype_from_str(D['dtype'], T)
                return new_dict

            return [_massage_dtype_in_dict(D) for D in A]


        T = self.type_dict()
        cube.register_properties(_massage_dtypes(P, T))
        cube.register_arrays(_massage_dtypes(A, T))

        #==========================================
        # Tensorflow Session and Thread Coordinator
        #==========================================

        # Create the tensorflow session object
        # Use supplied target, if present
        tf_server_target = slvr_cfg.get('tf_server_target', '')
        self._tf_session = tf.Session(tf_server_target)

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
            for n, a in cube.arrays().iteritems()
            if not a.temporary }

        montblanc.log.info("Data source '{dfs}'".format(dfs=data_source))

        # The descriptor queue items are not user-defined arrays
        # but a variable passed through describing a chunk of the
        # problem. Make it look as if it's an array
        if 'descriptor' in dfs:
            raise KeyError("'descriptor' is reserved, "
                "please use another array name.")

        dfs['descriptor'] = DataSource(lambda c: np.int32([0]), np.int32, 'Internal')

        #=======================
        # Data Sources and Sinks
        #=======================

        # Construct list of data sources and sinks
        # internal to the solver.
        # These will be overridden by source and sink
        # providers supplied by the user in the solve()
        # method
        self._source_providers = []
        self._sink_providers = [NullSinkProvider()]

        #==================
        # Data Source Cache
        #==================

        self._source_cache = {}
        self._source_cache_lock = threading.Lock()

        #==================
        # Memory Budgeting
        #==================

        # For deciding whether to rebudget
        self._previous_budget = 0

        #================
        # Cube Transcoder
        #================
        self._iter_dims = ['ntime', 'nbl']
        self._transcoder = CubeDimensionTranscoder(self._iter_dims)

        #=========================
        # Tensorflow devices
        #=========================

        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()

        gpus = [d.name for d in devices if d.device_type == 'GPU']
        cpus = [d.name for d in devices if d.device_type == 'CPU']

        self._devices = cpus if len(gpus) == 0 else gpus
        self._shards_per_device = spd = 3
        self._nr_of_shards = shards = len(self._devices)*spd
        # shard_id == d*ndev + shard
        shard = lambda d, s: d*len(self._devices)+s

        assert len(self._devices) > 0

        #=========================
        # Tensorflow Compute Graph
        #=========================

        # Create all tensorflow constructs within the compute graph
        with tf.Graph().as_default() as compute_graph:
            # Create our data feeding structure containing
            # input/output queues and feed once variables
            self._tf_feed_data = _construct_tensorflow_feed_data(
                dfs, cube, self._iter_dims, shards)

            # Construct tensorflow expressions for each shard
            self._tf_expr = [_construct_tensorflow_expression(
                    self._tf_feed_data, dev, shard(d,s))
                for d, dev in enumerate(self._devices)
                for s in range(self._shards_per_device)]

            # Initialisation operation
            init_op = tf.global_variables_initializer()
            # Now forbid modification of the graph
            compute_graph.finalize()

        #==========================================
        # Tensorflow Session
        #==========================================

        montblanc.log.debug("Attaching session to tensorflow server "
            "'{tfs}'".format(tfs=tf_server_target))

        session_config = tf.ConfigProto(allow_soft_placement=True)

        self._tf_session = tf.Session(tf_server_target,
            graph=compute_graph, config=session_config)
        self._tf_session.run(init_op)

        #======================
        # Thread pool executors
        #======================

        tpe = cf.ThreadPoolExecutor

        self._parameter_executor = tpe(1)
        self._feed_executors = [tpe(1) for i in range(shards)]
        self._compute_executors = [tpe(1) for i in range(shards)]
        self._consumer_executor = tpe(1)

    def _parameter_feed(self):
        try:
            self._parameter_feed_impl()
        except Exception as e:
            montblanc.log.exception("Parameter Exception")
            raise

    def _parameter_feed_impl(self):
        session = self._tf_session

        # Copy dimensions of the main cube
        cube = self.hypercube.copy()
        LQ = self._tf_feed_data.local

        # Get space of iteration
        iter_args = _iter_args(self._iter_dims, cube)
        parameters_fed = 0

        # Iterate through the hypercube space
        for i, d in enumerate(cube.dim_iter(*iter_args)):
            cube.update_dimensions(d)
            descriptor = self._transcoder.encode(cube.dimensions(copy=False))
            feed_dict = {LQ.parameter.placeholders[0] : descriptor }
            montblanc.log.debug('Encoding {i} {d}'.format(i=i, d=descriptor))
            session.run(LQ.parameter.enqueue_op, feed_dict=feed_dict)
            parameters_fed += 1

        montblanc.log.info("Done feeding {n} parameters.".format(
            n=parameters_fed))

        feed_dict = {LQ.parameter.placeholders[0] : [-1] }
        session.run(LQ.parameter.enqueue_op, feed_dict=feed_dict)

    def _feed(self, cube, data_sources, data_sinks, global_iter_args):
        """ Feed stub """
        try:
            self._feed_impl(cube, data_sources, data_sinks, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_impl(self, cube, data_sources, data_sinks, global_iter_args):
        """ Implementation of queue feeding """
        session = self._tf_session
        FD = self._tf_feed_data
        LQ = FD.local

        # Get source strides out before the local sizes are modified during
        # the source loops below
        src_types = LQ.src_queues.keys()
        src_strides = [int(i) for i in cube.dim_extent_size(*src_types)]
        src_queues = [[LQ.src_queues[t][s] for t in src_types]
            for s in range(self._nr_of_shards)]

        compute_feed_dict = { ph: cube.dim_global_size(n) for
            n, ph in FD.src_ph_vars.iteritems() }
        compute_feed_dict.update({ ph: getattr(cube, n) for
            n, ph in FD.property_ph_vars.iteritems() })

        chunks_fed = 0

        which_shard = itertools.cycle(range(self._nr_of_shards))

        while True:
            try:
                # Get the descriptor describing a portion of the RIME
                # and the current sizes of the the input queues
                ops = [LQ.parameter.dequeue_op] + [iq.size_op
                    for iq in LQ.input]
                result = session.run(ops)
                descriptor = result[0]
                input_queue_sizes = np.asarray(result[1:])
            except tf.errors.OutOfRangeError as e:
                montblanc.log.exception("Descriptor reading exception")

            # Quit if EOF
            if descriptor[0] == -1:
                break

            # Make it read-only so we can hash the contents
            descriptor.flags.writeable = False

            # Find indices of the emptiest queues and, by implication
            # the shard with the least work assigned to it
            emptiest_queues = np.argsort(input_queue_sizes)
            shard = emptiest_queues[0]
            shard = which_shard.next()

            feed_f = self._feed_executors[shard].submit(self._feed_actual,
                data_sources.copy(), cube.copy(),
                descriptor, shard,
                src_types, src_strides, src_queues[shard],
                global_iter_args)

            compute_f = self._compute_executors[shard].submit(self._compute,
                compute_feed_dict, shard)

            consume_f = self._consumer_executor.submit(self._consume,
                data_sinks.copy(), cube.copy(), global_iter_args)

            yield (feed_f, compute_f, consume_f)

        montblanc.log.info("Done feeding {n} chunks.".format(n=chunks_fed))

    def _feed_actual(self, *args):
        try:
            return self._feed_actual_impl(*args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_actual_impl(self, data_sources, cube,
            descriptor, shard,
            src_types, src_strides, src_queues,
            global_iter_args):

        session = self._tf_session
        iq = self._tf_feed_data.local.input[shard]

        # Decode the descriptor and update our cube dimensions
        dims = self._transcoder.decode(descriptor)
        cube.update_dimensions(dims)

        # Determine array shapes and data types for this
        # portion of the hypercube
        array_schemas = cube.arrays(reify=True)

        # Inject a data source and array schema for the
        # descriptor queue items. These aren't full on arrays per se
        # but they need to work within the feeding framework
        array_schemas['descriptor'] = descriptor
        data_sources['descriptor'] = DataSource(
            lambda c: descriptor, np.int32, 'Internal')

        # Generate (name, placeholder, datasource, array schema)
        # for the arrays required by each queue
        gen = ((a, ph, data_sources[a], array_schemas[a])
            for ph, a in zip(iq.placeholders, iq.fed_arrays))

        # Get input data by calling the data source functors
        input_data = [(a, ph, _get_data(ds, SourceContext(a, cube,
                self.config(), global_iter_args,
                cube.array(a) if a in cube.arrays() else {},
                ad.shape, ad.dtype)))
            for (a, ph, ds, ad) in gen]

        # Create a feed dictionary from the input data
        feed_dict = { ph: data for (a, ph, data) in input_data }

        # Cache the inputs for this chunk of data,
        # so that sinks can access them
        input_cache = { a: data for (a, ph, data) in input_data }

        # Guard access to the cache with a lock
        with self._source_cache_lock:
            self._source_cache[descriptor.data] = input_cache

        montblanc.log.info("Enqueueing chunk {d} on shard {sh}".format(
            d=descriptor, sh=shard))

        session.run(iq.enqueue_op, feed_dict=feed_dict)

        # For each source type, feed that source queue
        for src_type, queue, stride in zip(src_types, src_queues, src_strides):
            iter_args = [(src_type, stride)]

            # Iterate over chunks of the source
            for chunk_i, dim_desc in enumerate(cube.dim_iter(*iter_args)):
                cube.update_dimensions(dim_desc)
                s = dim_desc[0]['upper_extent'] - dim_desc[0]['lower_extent']


                montblanc.log.info("'{ci}: Enqueueing {d} '{s}' '{t}' sources "
                    "on shard {sh}".format(d=descriptor,
                        ci=chunk_i, s=s, t=src_type, sh=shard))

                # Determine array shapes and data types for this
                # portion of the hypercube
                array_schemas = cube.arrays(reify=True)

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

    def _compute(self, feed_dict, shard):
        """ Call the tensorflow compute """

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
        #    timeout_in_ms=10000)
        # run_options =tf.RunOptions()
        # run_metadata = tf.RunMetadata()

        try:
            descriptor, enq = self._tf_session.run(
                self._tf_expr[shard],
                feed_dict=feed_dict)
                #options=run_options,
                #run_metadata=run_metadata)

        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise

        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('compute-timeline.json', 'w') as f:
        #     f.write(ctf)

    def _consume(self, data_sinks, cube, global_iter_args):
        """ Consume stub """
        try:
            return self._consume_impl(data_sinks, cube, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Consumer Exception")
            raise e, None, sys.exc_info()[2]

    def _consume_impl(self, data_sinks, cube, global_iter_args):
        """ Consume """

        LQ = self._tf_feed_data.local
        output = self._tf_session.run(LQ.output.dequeue_op)

        # Expect the descriptor in the first tuple position
        assert len(output) > 0
        assert LQ.output.fed_arrays[0] == 'descriptor'

        descriptor = output[0]
        # Make it read-only so we can hash the contents
        descriptor.flags.writeable = False

        dims = self._transcoder.decode(descriptor)
        cube.update_dimensions(dims)

        # Obtain and remove input data from the source cache
        with self._source_cache_lock:
            try:
                input_data = self._source_cache.pop(descriptor.data)
            except KeyError:
                raise ValueError("No input data cache available "
                    "in source cache for descriptor {}!"
                        .format(descriptor))

        # For each array in our output, call the associated data sink
        for n, a in zip(LQ.output.fed_arrays[1:], output[1:]):
            sink_context = SinkContext(n, cube,
                self.config(), global_iter_args,
                cube.array(n) if n in cube.arrays() else {},
                a, input_data)

            _supply_data(data_sinks[n], sink_context)

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
        bytes_required = _apply_source_provider_dim_updates(self.hypercube,
            source_providers)

        # If we use more memory than previously,
        # perform another budgeting operation
        # to make sure everything fits
        if bytes_required > self._previous_budget:
            self._previous_budget = _budget(self.hypercube, self.config())

        #===================================
        # Assign data to Feed Once variables
        #===================================

        # Copy the hypercube
        cube = self.hypercube.copy()
        global_iter_args = _iter_args(self._iter_dims, cube)
        array_schemas = cube.arrays(reify=True)

        # Construct data sources from those supplied by the
        # source providers, if they're associated with
        # input sources
        LQ = self._tf_feed_data.local
        input_sources = LQ.input_sources
        data_sources = self._default_data_sources.copy()
        data_sources.update({n: DataSource(f, cube.array(n).dtype, prov.name())
            for prov in source_providers
            for n, f in prov.sources().iteritems()
            if n in input_sources})

        # Get data sinks from supplied providers
        data_sinks = { n: DataSink(f, prov.name())
            for prov in sink_providers
            for n, f in prov.sinks().iteritems()
            if not n == 'descriptor' }

        # Construct a feed dictionary from data sources
        feed_dict = {  fo.ph: _get_data(data_sources[k],
                SourceContext(k, cube,
                    self.config(), global_iter_args,
                    cube.array(k) if k in cube.arrays() else {},
                    array_schemas[k].shape,
                    array_schemas[k].dtype))
            for k, fo
            in LQ.feed_once.iteritems() }

        # Run the assign operations for each feed_once variable
        self._tf_session.run([fo.assign_op for fo in LQ.feed_once.itervalues()],
            feed_dict=feed_dict)

        try:
            params = self._parameter_executor.submit(self._parameter_feed)
            not_done = set([params])

            for futures in self._feed_impl(cube,
                data_sources, data_sinks, global_iter_args):

                not_done.update(futures)

            # TODO: timeout not strictly necessary
            done, not_done = cf.wait(not_done,
                return_when=cf.ALL_COMPLETED)

            for future in done:
                res = future.result()

            assert len(not_done) == 0

        except (KeyboardInterrupt, SystemExit) as e:
            montblanc.log.exception('Solving interrupted')
            raise
        except:
            montblanc.log.exception('Solving exception')

    def close(self):
        # Shutdown thread executors
        self._parameter_executor.shutdown()
        [fe.shutdown() for fe in self._feed_executors]
        [ce.shutdown() for ce in self._compute_executors]
        self._consumer_executor.shutdown()

        # Shutdown thte tensorflow session
        self._tf_session.close()

        # Shutdown data sources
        for source in self._source_providers:
            source.close()

        # Shutdown data sinks
        for sink in self._sink_providers:
            sink.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()


def _construct_tensorflow_feed_data(dfs, cube, iter_dims,
    nr_of_input_queues):

    QUEUE_SIZE = 10

    FD = AttrDict()
    # https://github.com/bcj/AttrDict/issues/34
    FD._setattr('_sequence_type', list)
    # Reference local queues
    FD.local = local = AttrDict()
    # https://github.com/bcj/AttrDict/issues/34
    local._setattr('_sequence_type', list)

    # Create placholder variables for source counts
    FD.src_ph_vars = AttrDict({
        n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
        for n in ['nsrc'] + mbu.source_nr_vars()})

    # Create placeholder variables for properties
    FD.property_ph_vars = AttrDict({
        n: tf.placeholder(dtype=p.dtype, shape=(), name=n)
        for n, p in cube.properties().iteritems() })

    #========================================================
    # Determine which arrays need feeding once/multiple times
    #========================================================

    # Take all arrays flagged as input
    input_arrays = [a for a in cube.arrays().itervalues()
                    if a.get('input', False) == True]

    def _partition(pred, iterable):
        t1, t2 = itertools.tee(iterable)
        return (itertools.ifilterfalse(pred, t1),
            itertools.ifilter(pred, t2))

    # Convert to set
    iter_dims = set(iter_dims)

    iterating = lambda a: len(iter_dims.intersection(a.shape)) > 0
    feed_once, feed_all = _partition(iterating, input_arrays)
    feed_once, feed_all = list(feed_once), list(feed_all)

    #======================================
    # Create tensorflow queues which
    # require feeding multiple times
    #======================================

    # Create the queue feeding parameters into the system
    local.parameter = create_queue_wrapper('descriptors',
        QUEUE_SIZE, ['descriptor'], dfs)

    # Create the queue for holding the input
    local.input = [create_queue_wrapper('input_%d' % i, QUEUE_SIZE,
                ['descriptor'] + [a.name for a in feed_all], dfs)
            for i in range(nr_of_input_queues)]

    # Create source input queues
    local.point_source = [create_queue_wrapper('point_source_%d' % i,
        QUEUE_SIZE, ['point_lm', 'point_stokes', 'point_alpha'], dfs)
        for i in range(nr_of_input_queues)]

    local.gaussian_source = [create_queue_wrapper('gaussian_source_%d' % i,
        QUEUE_SIZE, ['gaussian_lm', 'gaussian_stokes', 'gaussian_alpha',
            'gaussian_shape'], dfs)
        for i in range(nr_of_input_queues)]

    local.sersic_source = [create_queue_wrapper('sersic_source_%d' % i,
        QUEUE_SIZE, ['sersic_lm', 'sersic_stokes', 'sersic_alpha',
            'sersic_shape'], dfs)
        for i in range(nr_of_input_queues)]

    # Source queues to feed
    local.src_queues = src_queues = {
        'npsrc' : local.point_source,
        'ngsrc' : local.gaussian_source,
        'nssrc' : local.sersic_source,
    }

    #======================================
    # The single output queue
    #======================================

    local.output = create_queue_wrapper('output', QUEUE_SIZE,
        ['descriptor', 'model_vis'], dfs)

    #=================================================
    # Create tensorflow variables which are
    # fed only once via an assign operation
    #=================================================

    def _make_feed_once_tuple(array):
        dtype = dfs[array.name].dtype

        ph = tf.placeholder(dtype=dtype,
            name=a.name + "_placeholder")

        var = tf.Variable(tf.zeros(shape=(1,), dtype=dtype),
            validate_shape=False,
            name=array.name)

        op = tf.assign(var, ph, validate_shape=False)
        #op = tf.Print(op, [tf.shape(var), tf.shape(op)],
        #    message="Assigning {}".format(array.name))

        return FeedOnce(ph, var, op)

    # Create placeholders, variables and assign operators
    # for data sources that we will only feed once
    local.feed_once = { a.name : _make_feed_once_tuple(a)
        for a in feed_once }

    #=======================================================
    # Construct the list of data sources that need feeding
    #=======================================================

    # Data sources from input queues
    input_queues = local.input + [q for sq in src_queues.values() for q in sq]
    input_sources = { a for q in input_queues
        for a in q.fed_arrays}

    # Data sources from feed once variables
    input_sources.update(local.feed_once.keys())

    local.input_sources = input_sources

    return FD

def _construct_tensorflow_expression(feed_data, device, shard):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)
    src_count = zero
    src_ph_vars = feed_data.src_ph_vars

    LQ = feed_data.local

    # Pull RIME inputs out of the feed queue
    # of the relevant shard
    D = LQ.input[shard].dequeue_to_attrdict()
    D.update({k: fo.var for k, fo in LQ.feed_once.iteritems()})

    # Infer chunk dimensions
    with tf.device(device):
        model_vis_shape = tf.shape(D.model_vis)
        ntime, nbl, nchan = [model_vis_shape[i] for i in range(3)]
        FT, CT = D.uvw.dtype, D.model_vis.dtype

    def apply_dies(src_count):
        """ Have we reached the maximum source count """
        return tf.greater_equal(src_count, src_ph_vars.nsrc)

    def antenna_jones(lm, stokes, alpha):
        """
        Compute the jones terms for each antenna.

        lm, stokes and alpha are the source variables.
        """
        cplx_phase = rime.phase(lm, D.uvw, D.frequency, CT=CT)

        bsqrt, sgn_brightness = rime.b_sqrt(stokes, alpha,
            D.frequency, D.ref_frequency, CT=CT)

        ejones = rime.e_beam(lm, D.frequency,
            D.point_errors, D.antenna_scaling,
            D.parallactic_angles,
            D.beam_extents, D.beam_freq_map, D.ebeam)

        return (rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=FT),
            sgn_brightness)

    # While loop condition for each point source type
    def point_cond(model_vis, npsrc, src_count):
        return tf.less(npsrc, src_ph_vars.npsrc)

    def gaussian_cond(model_vis, ngsrc, src_count):
        return tf.less(ngsrc, src_ph_vars.ngsrc)

    def sersic_cond(model_vis, nssrc, src_count):
        return tf.less(nssrc, src_ph_vars.nssrc)

    # While loop bodies
    def point_body(model_vis, npsrc, src_count):
        """ Accumulate visiblities for point source batch """
        lm, stokes, alpha = LQ.point_source[shard].dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        npsrc +=  nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=FT)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, D.flag, D.gterm,
            model_vis, apply_dies(src_count))

        return model_vis, npsrc, src_count

    def gaussian_body(model_vis, ngsrc, src_count):
        """ Accumulate visiblities for gaussian source batch """
        lm, stokes, alpha, gauss_params = LQ.gaussian_source[shard].dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        ngsrc += nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        gauss_shape = rime.gauss_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, gauss_params)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            gauss_shape, ant_jones, sgn_brightness, D.flag, D.gterm,
            model_vis, apply_dies(src_count))

        return model_vis, ngsrc, src_count

    def sersic_body(model_vis, nssrc, src_count):
        """ Accumulate visiblities for sersic source batch """
        lm, stokes, alpha, sersic_params = LQ.sersic_source[shard].dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        nssrc += nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        sersic_shape = rime.sersic_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, sersic_params)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            sersic_shape, ant_jones, sgn_brightness, D.flag, D.gterm,
            model_vis, apply_dies(src_count))

        return model_vis, nssrc, src_count

    with tf.device(device):
        # Evaluate point sources
        model_vis, npsrc, src_count = tf.while_loop(
            point_cond, point_body,
            [D.model_vis, zero, src_count])

        # Evaluate gaussians
        model_vis, ngsrc, src_count = tf.while_loop(
            gaussian_cond, gaussian_body,
            [model_vis, zero, src_count])

        # Evaluate sersics
        model_vis, nssrc, src_count = tf.while_loop(
            sersic_cond, sersic_body,
            [model_vis, zero, src_count])

    # Create enqueue operation
    enqueue_op = LQ.output.queue.enqueue([D.descriptor, model_vis])

    # Return descriptor and enqueue operation
    return D.descriptor, enqueue_op

def _get_data(data_source, context):
    """ Get data from the data source, checking the return values """
    try:
        # Get data from the data source
        data = data_source.source(context)

        # Complain about None values
        if data is None:
            raise ValueError("'None' returned from "
                "data source '{n}'".format(n=context.name))
        # We want numpy arrays
        elif not isinstance(data, np.ndarray):
            raise TypeError("Data source '{n}' did not "
                "return a numpy array, returned a '{t}'".format(
                    t=type(data)))
        # And they should be the right shape and type
        elif data.shape != context.shape or data.dtype != context.dtype:
            raise ValueError("Expected data of shape '{esh}' and "
                "dtype '{edt}' for data source '{n}', but "
                "shape '{rsh}' and '{rdt}' was found instead".format(
                    n=context.name, esh=context.shape, edt=context.dtype,
                    rsh=data.shape, rdt=data.dtype))

        return data

    except Exception as e:
        ex = ValueError("An exception occurred while "
            "obtaining data from data source '{ds}'\n\n"
            "{e}\n\n"
            "{help}".format(ds=context.name,
                e=str(e), help=context.help()))

        raise ex, None, sys.exc_info()[2]

def _supply_data(data_sink, context):
    """ Supply data to the data sink """
    try:
        data_sink.sink(context)
    except Exception as e:
        ex = ValueError("An exception occurred while "
            "supplying data to data sink '{ds}'\n\n"
            "{e}\n\n"
            "{help}".format(ds=context.name,
                e=str(e), help=context.help()))

        raise ex, None, sys.exc_info()[2]

def _iter_args(iter_dims, cube):
    iter_strides = cube.dim_extent_size(*iter_dims)
    return zip(iter_dims, iter_strides)

def _uniq_log2_range(start, size, div):
    start = np.log2(start)
    size = np.log2(size)
    int_values = np.int32(np.logspace(start, size, div, base=2)[:-1])

    return np.flipud(np.unique(int_values))

BUDGETING_DIMS = ['ntime', 'nbl', 'nsrc'] + mbu.source_nr_vars()

def _budget(cube, slvr_cfg):
    # Figure out a viable dimension configuration
    # given the total problem size
    mem_budget = slvr_cfg.get('mem_budget', 2*ONE_GB)
    bytes_required = cube.bytes_required()

    src_dims = mbu.source_nr_vars() + [Options.NSRC]
    dim_names = ['na', 'nbl', 'ntime'] + src_dims
    global_sizes = cube.dim_global_size(*dim_names)
    na, nbl, ntime = global_sizes[:3]

    # Keep track of original dimension sizes and any reductions that are applied
    original_sizes = { r: s for r, s in zip(dim_names, global_sizes) }
    applied_reductions = {}

    def _reduction():
        # Reduce over time first
        trange = _uniq_log2_range(1, ntime, 5)
        for t in trange[0:1]:
            yield [('ntime', t)]

        # Attempt reduction over source
        sbs = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)
        srange = _uniq_log2_range(10, sbs, 5) if sbs > 10 else 10
        src_dim_gs = global_sizes[3:]

        for bs in srange:
            yield [(d, bs if bs < gs else gs) for d, gs
                in zip(src_dims, src_dim_gs)]

        # Try the rest of the timesteps
        for t in trange[1:]:
            yield [('ntime', t)]

        # Reduce by baseline
        for bl in _uniq_log2_range(na, nbl, 5):
            yield [('nbl', bl)]

    for reduction in _reduction():
        if bytes_required > mem_budget:
            for dim, size in reduction:
                applied_reductions[dim] = size
                cube.update_dimension(dim, lower_extent=0, upper_extent=size)
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

    def _transform_update(d):
        if isinstance(d, tuple):
            return HyperCubeDim(d[0], d[1])
        elif isinstance(d, HyperCubeDim):
            return d
        else:
            raise TypeError("Expected a hypercube dimension or "
                "('dim_name', dim_size) tuple "
                "for dimension update. Instead received "
                "'{d}'".format(d=d))



    # Update the mapping, except for the nsrc dimension
    [mapping[d.name].append((d, prov.name()))
        for prov in source_providers
        for d in (_transform_update(d) for d in prov.updated_dimensions())
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

    # Get existing dimension extents
    extent_sizes = cube.dim_extent_size(*mapping.keys())

    # Now update our dimensions
    for (n, u), es in zip(mapping.iteritems(), extent_sizes):
        # Reduce our local size to satisfy hypercube
        d = u[0][0]
        gs = d.global_size
        # Defer to existing extent size for budgeting dimensions
        es = es if es in BUDGETING_DIMS else d.extent_size
        # Clamp extent size to global size
        es = gs if es > gs else es
        cube.update_dimension(n, global_size=gs,
            lower_extent=0, upper_extent=es)

    # Handle global number of sources differently
    # It's equal to the number of
    # point's, gaussian's, sersic's combined
    nsrc = sum(cube.dim_global_size(*mbu.source_nr_vars()))

    # Local number of sources will be the extent size of whatever
    # source type we're currently iterating over. So just take
    # the maximum extent size given the sources
    ls = max(cube.dim_extent_size(*mbu.source_nr_vars()))

    cube.update_dimension('nsrc', global_size=nsrc,
        lower_extent=0, upper_extent=ls)

    # Return our cube size
    return cube.bytes_required()

