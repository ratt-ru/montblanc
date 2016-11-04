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
import json
import threading
import time
import sys

from attrdict import AttrDict
import concurrent.futures as cf
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

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
import hypercube.util as hcu

from montblanc.solvers import MontblancTensorflowSolver
from montblanc.config import RimeSolverConfig as Options

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

rime = load_tf_lib()

DataSource = collections.namedtuple("DataSource", ['source', 'dtype', 'name'])
DataSink = collections.namedtuple("DataSink", ['sink', 'name'])
FeedOnce = collections.namedtuple("FeedOnce", ['ph', 'var', 'assign_op'])

class AConfigurationStrategy(object):
    def construct_shared_configuration(self, slvr_cfg):
        raise NotImplementedError()

    def transmit_receive_config(self, session, tf_dict, slvr_cfg):
        raise NotImplementedError()

class ConfigurationStrategy(AConfigurationStrategy):
    def __init__(self, tf_server_target, job_name, task_index):
        self._target = tf_server_target
        self._job = job_name
        self._task = task_index
        self._dev_spec = tf.DeviceSpec(job_name, task_index)

    def _get_tf_dict_and_cfg_graph(self, slvr_cfg):
        # Construct a graph for transmission/reception of configuration
        cfg_graph = tf.Graph()

        with cfg_graph.as_default():
            D = self.construct_shared_configuration(slvr_cfg)

            # Create placholders for enqueue operations
            D.local_dev_ph = tf.placeholder(tf.string)
            D.gpu_dev_ph = tf.placeholder(tf.string)
            D.gpu_mem_size_ph = tf.placeholder(tf.int32)

            # Construct some ops for interacting with the
            # shared queue
            D.gpu_enqueue_op = D.gpu_queue.enqueue(
                [D.local_dev_ph, D.gpu_dev_ph, D.gpu_mem_size_ph])
            D.gpu_dequeue_op = D.gpu_queue.dequeue()
            D.gpu_size_op = D.gpu_queue.size()

        montblanc.log.debug("Created configuration graph")

        return D, cfg_graph

    def execute(self, slvr_cfg):
        D, cfg_graph = self._get_tf_dict_and_cfg_graph(slvr_cfg)

        # Create session for transmission/reception of configuration
        with tf.Session(self._target, graph=cfg_graph) as S:
            return self.transmit_receive_config(S, D, slvr_cfg)

    def _create_shared_config_internal(self, configuration):
        """
        Create shared configuration state
        in the 'shared' container
        """
        D = AttrDict()

        with tf.container('shared'):
            D.gpu_queue = tf.FIFOQueue(1000,
                [tf.string, tf.string, tf.int32], [(), (), ()],
                name="gpu_queue", shared_name="shared_gpu_queue")
            D.shared_cfg = tf.Variable(configuration, name="master_configuration")

        return D

class WorkerConfigurationStrategy(ConfigurationStrategy):
    def __init__(self, *args):
        super(WorkerConfigurationStrategy, self).__init__(*args)

    def construct_shared_configuration(self, slvr_cfg):
        # Pass in dummy configuration string
        return self._create_shared_config_internal("dummy")

    def transmit_receive_config(self, session, tf_dict, slvr_cfg):
        montblanc.log.debug("Transmitting GPU configuration")
        from tensorflow.python.client import device_lib

        gpus = [x for x in device_lib.list_local_devices()
            if x.device_type == 'GPU']

        for gpu in gpus:
            # If the session is attaching to a tf.train.Server.target,
            # the server will have grabbed the GPU memory. So
            # these memory
            montblanc.log.debug("Enqueuing {n} with size {s}".format(
                n=gpu.name, s=gpu.memory_limit))
            session.run(tf_dict.gpu_enqueue_op, feed_dict={
                tf_dict.local_dev_ph : self._dev_spec.to_string(),
                tf_dict.gpu_dev_ph : gpu.name,
                tf_dict.gpu_mem_size_ph : gpu.memory_limit,
            })

        montblanc.log.debug("GPU configuration transmitted")

        montblanc.log.debug("Receiving shared configuration")

        # The server may not have initialized this variable
        # Wait a bit for some iterations
        for i in xrange(5):
            if not session.run(tf.is_variable_initialized(tf_dict.shared_cfg)):
                montblanc.log.debug("{i} Waiting a second for "
                    "server configuration".format(i=i))
                time.sleep(1)

        slvr_cfg = json.loads(session.run(tf_dict.shared_cfg))
        montblanc.log.debug("Received shared configuration")
        # Replace job name and task index with local information
        slvr_cfg['tf_job_name'] = self._job
        slvr_cfg['tf_task_index'] = self._task

        return slvr_cfg

class MasterConfigurationStrategy(ConfigurationStrategy):
    def __init__(self, *args):
        super(MasterConfigurationStrategy, self).__init__(*args)

    def construct_shared_configuration(self, slvr_cfg):
        with tf.device(self._dev_spec):
            cfg_str = json.dumps(slvr_cfg)
            return self._create_shared_config_internal(cfg_str)

    def transmit_receive_config(self, session, tf_dict, slvr_cfg):
        montblanc.log.debug("Initialising shared configuration")
        session.run(tf.initialize_variables([tf_dict.shared_cfg]))
        montblanc.log.debug("Done initialising shared configuration")

        #time.sleep(1)

        montblanc.log.debug("Reading worker GPU configuration")
        # Now read the gpu configuration from each node
        while session.run(tf_dict.gpu_size_op) > 0:
            _local_dev, _gpu_name, _gpu_mem = session.run(tf_dict.gpu_dequeue_op)
            montblanc.log.info("Found gpu {g} "
                "with size {s} on device {d}".format(
                    d=_local_dev, g=_gpu_name, s=hcu.fmt_bytes(_gpu_mem,)))

        montblanc.log.debug("Worker GPU configuration read")

        return slvr_cfg

class RimeSolver(MontblancTensorflowSolver):
    """ RIME Solver Implementation """

    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """

        #==============================================
        # Obtain tensorflow server target, job and task
        #==============================================

        try:
            server, job, task = (slvr_cfg[n] for n in ('tf_server_target',
                                                        'tf_job_name',
                                                        'tf_task_index'))
        except KeyError as e:
            msg = "'%s' missing from solver configuration!" % e.message
            raise (KeyError(msg), None, sys.exc_info()[2])

        self._is_master = job == "master" and task == 0

        #===============================================
        # Transmit/Receive master/worker configuration
        #===============================================

        Strat = (MasterConfigurationStrategy if self._is_master
            else WorkerConfigurationStrategy)

        slvr_cfg = Strat(server, job, task).execute(slvr_cfg)

        # Extract the cluster definition
        cluster = slvr_cfg['tf_cluster']

        #=============================================
        # Defer to parent construct with configuration
        #=============================================

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

        # Construct list of data sources/sinks internal to the solver
        # Any data sources/sinks specified in the solve() method will
        # override these
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

        #======================
        # Thread pool executors
        #======================

        self._parameter_executor = cf.ThreadPoolExecutor(1)
        self._feed_executor = cf.ThreadPoolExecutor(1)
        self._compute_executor = cf.ThreadPoolExecutor(1)
        self._consumer_executor = cf.ThreadPoolExecutor(1)

        #=========================
        # Tensorflow Compute Graph
        #=========================

        # Create all tensorflow constructs within the compute graph
        with tf.Graph().as_default() as compute_graph:
            # Create feed data and expression
            self._tf_feed_data = _construct_tensorflow_feed_data(dfs,
                cube, cluster, job, task, self._iter_dims)
            self._tf_expr = _construct_tensorflow_expression(
                self._tf_feed_data)

            # Initialisation operation
            init_op = tf.initialize_local_variables()

            # Now forbid modification of the graph
            compute_graph.finalize()

        #==========================================
        # Tensorflow Session
        #==========================================

        self._tf_server = server
        self._tf_job = job
        self._tf_task = task

        self._tf_coord = tf.train.Coordinator()

        montblanc.log.debug("Attaching session to tensorflow server "
            "'{tfs}'".format(tfs=server))
        self._tf_session = tf.Session(server, graph=compute_graph)
        self._tf_session.run(init_op)

    def is_master(self):
        return self._is_master

    def _parameter_feed(self):
        # Only the master feeds descriptors
        if not self.is_master():
            return

        try:
            self._parameter_feed_impl()
        except Exception as e:
            montblanc.log.exception("Parameter Exception")
            raise

    def _parameter_feed_impl(self):
        def _make_hashring(remote_parameter_queues):
            from uhashring import HashRing

            node_config = { '%s_%s' % (j,t) : {
                'weight': 100,
                'vnodes': 40,
                'hostname': j,
                'port': t }
                for j, t, q in remote_parameter_queues }

            return HashRing(node_config, replicas=4, compat=False)

        session = self._tf_session

        # Copy dimensions of the main cube
        cube = self.hypercube.copy()
        RPQ = self._tf_feed_data.remote.parameter

        hashring = _make_hashring(RPQ)

        # Get space of iteration
        iter_args = _iter_args(self._iter_dims, cube)
        parameters_fed = 0

        # Iterate through the hypercube space
        for i, d in enumerate(cube.dim_iter(*iter_args, update_local_size=True)):
            cube.update_dimensions(d)

            # Construct a descriptor describing a portion of the problem
            descriptor = self._transcoder.encode(cube.dimensions(copy=False))

            # Hash the descriptor to obtain the node to send it to
            node = hashring.get(descriptor)

            # Get the remote queue
            job, task, queue = RPQ[node['port']]

            # Feed the queue with the descriptor
            montblanc.log.debug('{i} Placing {d} on {ds}'.format(i=i,
                d=descriptor,
                ds=tf.DeviceSpec(job=job,task=task).to_string()))

            feed_dict = { queue.placeholders[0] : descriptor }
            session.run(queue.enqueue_op, feed_dict=feed_dict)
            parameters_fed += 1

        # Indicate EOF to workers
        for job, task, queue in RPQ:
            session.run(queue.enqueue_op, feed_dict={ queue.placeholders[0]: [-1] })

        self._tf_coord.request_stop()

        montblanc.log.info("Done feeding {n} parameters.".format(
            n=parameters_fed))

    def _feed(self, cube, data_sources, global_iter_args):
        """ Feed stub """

        # Only workers feed data
        if self.is_master():
            return

        try:
            self._feed_impl(cube, data_sources, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_impl(self, cube, data_sources, global_iter_args):
        """ Implementation of queue feeding """
        session = self._tf_session
        LQ = self._tf_feed_data.local

        # Get source strides out before the local sizes are modified during
        # the source loops below
        src_types = LQ.src_queues.keys()
        src_strides = [int(i) for i in cube.dim_local_size(*src_types)]
        src_queues = [LQ.src_queues[t] for t in src_types]

        chunks_fed = 0

        while True:
            try:
                # Get the descriptor describing a portion of the RIME
                descriptor = session.run(LQ.parameter.dequeue_op)
            except tf.errors.OutOfRangeError as e:
                montblanc.log.exception("Descriptor reading exception")

            # Make it read-only so we can hash the contents
            descriptor.flags.writeable = False
            montblanc.log.info("Received descriptor {}".format(descriptor))

            if descriptor[0] == -1:
                break

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
            data_sources['descriptor'] = DataSource(lambda c: descriptor, np.int32, 'Internal')

            # Generate (name, placeholder, datasource, array schema)
            # for the arrays required by each queue
            iq = LQ.input
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

            montblanc.log.debug("Enqueueing chunk {i} {d}".format(
                i=chunks_fed, d=descriptor))

            session.run(LQ.input.enqueue_op, feed_dict=feed_dict)

            chunks_fed += 1

            # For each source type, feed that source queue
            for src_type, queue, stride in zip(src_types, src_queues, src_strides):
                iter_args = [(src_type, stride)]

                # Iterate over local_size chunks of the source
                for chunk_i, dim_desc in enumerate(cube.dim_iter(*iter_args, update_local_size=True)):
                    cube.update_dimensions(dim_desc)

                    montblanc.log.debug("'{ci}: Enqueueing '{s}' '{t}' sources".format(
                        ci=chunk_i, s=dim_desc[0]['local_size'], t=src_type))

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


        # Close all local queues
        # session.run([q.close_op for q in (
        #     [LQ.input] + LQ.src_queues.values() + [LQ.output])])

        montblanc.log.info("Done feeding {n} chunks.".format(n=chunks_fed))

    def _compute_impl(self):
        """ Implementation of computation """

        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
        #    timeout_in_ms=10000)
        run_options =tf.RunOptions()
        run_metadata = tf.RunMetadata()

        S = self._tf_session
        FD = self._tf_feed_data
        cube = self.hypercube

        chunks_computed = 0

        feed_dict = { ph: cube.dim_global_size(n) for
            n, ph in FD.src_ph_vars.iteritems() }

        feed_dict.update({ ph: getattr(cube, n) for
            n, ph in FD.property_ph_vars.iteritems() })

        while not self._tf_coord.should_stop():
            try:
                descriptor, enq = S.run(self._tf_expr,
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError) as e:
                self._tf_coord.request_stop()
                continue

            # Are we done?
            dims = self._transcoder.decode(descriptor)
            chunks_computed += 1

        montblanc.log.info("Done computing {n} chunks."
            .format(n=chunks_computed))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('compute-timeline.json', 'w') as f:
            f.write(ctf)

    def _compute(self):
        """ Compute stub """
        # Only workers compute
        if self.is_master():
            return

        try:
            return self._compute_impl()
        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise

    def _consume(self, sink_providers):
        """ Consume stub """
        # Only workers consume data
        if self.is_master():
            return

        try:
            return self._consume_impl(sink_providers)
        except Exception as e:
            montblanc.log.exception("Consumer Exception")
            raise

    def _consume_impl(self, sink_providers):
        """ Consume """

        S = self._tf_session
        chunks_consumed = 0

        # Maintain a hypercube based on the main cube
        cube = self.hypercube.copy()
        LQ = self._tf_feed_data.local

        # Get space of iteration
        global_iter_args = _iter_args(self._iter_dims, cube)

        # Get data sinks from supplied providers
        data_sinks = { n: DataSink(f, prov.name())
            for prov in sink_providers
            for n, f in prov.sinks().iteritems()
            if not n == 'descriptor' }


        while not self._tf_coord.should_stop():
            try:
                output = S.run(LQ.output.dequeue_op)
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError) as e:
                self._tf_coord.request_stop()
                continue

            chunks_consumed += 1

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

        montblanc.log.info('Done consuming {n} chunks'.format(n=chunks_consumed))

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


        # Construct a feed dictionary from data sources
        feed_dict = {  ph: _get_data(data_sources[k],
                SourceContext(k, cube,
                    self.config(), global_iter_args,
                    cube.array(k) if k in cube.arrays() else {},
                    array_schemas[k].shape,
                    array_schemas[k].dtype))
            for k, (ph, var, assign_op)
            in LQ.feed_once.iteritems() }

        # Run the assign operations for each feed_once variable
        self._tf_session.run([fo.assign_op for fo in LQ.feed_once.itervalues()],
            feed_dict=feed_dict)

        not_done = []

        try:
            params = self._parameter_executor.submit(self._parameter_feed)
            feed = self._feed_executor.submit(self._feed, cube, data_sources,
                global_iter_args)
            compute = self._compute_executor.submit(self._compute)
            consume = self._consumer_executor.submit(self._consume, sink_providers)

            not_done = [params, feed, compute, consume]

            while not self._tf_coord.should_stop():
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

        # Cancel any running futures
        [f.cancel() for f in not_done]

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

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()


def _construct_tensorflow_feed_data(dfs, cube, cluster,
        job, task, iter_dims):

    QUEUE_SIZE = 10

    FD = AttrDict()
    # Reference local and remote queues
    FD.local = local = AttrDict()
    FD.remote = remote = AttrDict()

    # Create placholder variables for source counts
    FD.src_ph_vars = AttrDict({
        n: tf.placeholder(dtype=tf.int32, shape=(), name=n)
        for n in ['nsrc'] + mbu.source_nr_vars()})

    # Create placeholder variables for properties
    FD.property_ph_vars = AttrDict({
        n: tf.placeholder(dtype=p.dtype, shape=(), name=n)
        for n, p in cube.properties().iteritems() })

    local.parameter = None
    remote.parameter = parameter = []

    is_worker = job == 'worker'
    is_master = job == 'master' and task == 0

    pqn = lambda j, t: '_'.join([j, str(t), 'parameter_queue'])
    mkq = lambda sn: create_queue_wrapper('descriptors',
        QUEUE_SIZE, ['descriptor'], dfs, shared_name=sn)
    dev_spec = tf.DeviceSpec(job=job, task=task)

    if is_worker:

        # If this a worker, create the queue receiving parameters
        montblanc.log.info("Creating parameter queue on {ds}".format(
            ds=dev_spec.to_string()))

        with tf.device(dev_spec), tf.container('shared'):
            shared_name = pqn(job, task)
            local.parameter = mkq(shared_name)

    elif is_master:

        montblanc.log.info("Accessing remote parameter queue")
        montblanc.log.info("Remote parameters {}".format(remote.parameter))

        wjob = 'worker'
        nworkers = len(cluster[wjob])

        with tf.container('shared'):
            for t in xrange(nworkers):
                shared_name = pqn(wjob, t)
                wdev_spec = tf.DeviceSpec(job=wjob, task=t)
                montblanc.log.info("Accessing queue {}".format(shared_name))

                with tf.device(wdev_spec):
                    parameter.append((wjob, t, mkq(shared_name)))

        montblanc.log.info("Remote parameters {}".format(remote.parameter))

        for j ,t, q in remote.parameter:
            montblanc.log.info((j, t, q.queue.name))
    else:
        raise ValueError("Unhandled job/task pair ({},{})".format(job, task))

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

    with tf.device(dev_spec):
        # Create the queue for holding the input
        local.input = create_queue_wrapper('input', QUEUE_SIZE,
                    ['descriptor'] + [a.name for a in feed_all],
                    dfs)

        # Create source input queues
        local.point_source = create_queue_wrapper('point_source',
            QUEUE_SIZE, ['point_lm', 'point_stokes', 'point_alpha'], dfs)

        local.gaussian_source = create_queue_wrapper('gaussian_source',
            QUEUE_SIZE, ['gaussian_lm', 'gaussian_stokes', 'gaussian_alpha',
                'gaussian_shape'], dfs)

        local.sersic_source = create_queue_wrapper('sersic_source',
            QUEUE_SIZE, ['sersic_lm', 'sersic_stokes', 'sersic_alpha',
                'sersic_shape'], dfs)

        # Source queues to feed
        local.src_queues = src_queues = {
            'npsrc' : local.point_source,
            'ngsrc' : local.gaussian_source,
            'nssrc' : local.sersic_source,
        }

    #======================================
    # The single output queue
    #======================================

    with tf.device(dev_spec):
        local.output = create_queue_wrapper('output',
            QUEUE_SIZE, ['descriptor', 'model_vis'], dfs)

    #=================================================
    # Create tensorflow queues which are fed only once
    # via an assign operation
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
    with tf.device(dev_spec):
        local.feed_once = { a.name : _make_feed_once_tuple(a)
            for a in feed_once }

    #=======================================================
    # Construct the list of data sources that need feeding
    #=======================================================

    # Data sources from input queues
    input_sources = {a for q in [local.input] + src_queues.values()
        for a in q.fed_arrays}

    # Data sources from feed once variables
    input_sources.update(local.feed_once.keys())

    local.input_sources = input_sources

    return FD

def _construct_tensorflow_expression(feed_data):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)
    src_count = zero
    src_ph_vars = feed_data.src_ph_vars

    LQ = feed_data.local

    # Pull RIME inputs out of the feed queues
    D = LQ.input.dequeue_to_attrdict()
    D.update({k: fo.var for k, fo in LQ.feed_once.iteritems()})

    # Infer chunk dimensions
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

        return rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=FT), sgn_brightness

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
        lm, stokes, alpha = LQ.point_source.dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        npsrc +=  nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=FT)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, D.flag, D.gterm, model_vis,
            apply_dies(src_count))

        return model_vis, npsrc, src_count

    def gaussian_body(model_vis, ngsrc, src_count):
        """ Accumulate visiblities for gaussian source batch """
        lm, stokes, alpha, gauss_params = LQ.gaussian_source.dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        ngsrc += nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        gauss_shape = rime.gauss_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, gauss_params)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            gauss_shape, ant_jones, sgn_brightness, D.flag, D.gterm, model_vis,
            apply_dies(src_count))

        return model_vis, ngsrc, src_count

    def sersic_body(model_vis, nssrc, src_count):
        """ Accumulate visiblities for sersic source batch """
        lm, stokes, alpha, sersic_params = LQ.sersic_source.dequeue()
        nsrc = tf.shape(lm)[0]
        src_count += nsrc
        nssrc += nsrc
        ant_jones, sgn_brightness = antenna_jones(lm, stokes, alpha)
        sersic_shape = rime.sersic_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, sersic_params)
        model_vis = rime.sum_coherencies(D.antenna1, D.antenna2,
            sersic_shape, ant_jones, sgn_brightness, D.flag, D.gterm, model_vis,
            apply_dies(src_count))

        return model_vis, nssrc, src_count

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
            "{help}".format(e=str(e), ds=context.name, help=context.help()))

        raise ex, None, sys.exc_info()[2]

def _supply_data(data_sink, context):
    """ Supply data to the data sink """
    try:
        data_sink.sink(context)
    except Exception as e:
        ex = ValueError("An exception occurred while "
            "supplying data to data sink '{ds}'\n\n"
            "{help}".format(ds=context.name, help=context.help()))

        raise ex, None, sys.exc_info()[2]


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

    # Handle global number of sources differently
    # It's equal to the number of
    # point's, gaussian's, sersic's combined
    nsrc = sum(cube.dim_global_size(*mbu.source_nr_vars()))

    # Local number of sources will be the local size of whatever
    # source type we're currently iterating over. So just take
    # the maximum local size given the sources
    ls = max(cube.dim_local_size(*mbu.source_nr_vars()))

    cube.update_dimension('nsrc',
        local_size=ls, global_size=nsrc,
        lower_extent=0, upper_extent=ls)

    # Return our cube size
    return cube.bytes_required()

