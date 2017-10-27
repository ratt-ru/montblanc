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
import types

import concurrent.futures as cf
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from attrdict import AttrDict
import attr

import montblanc
import montblanc.util as mbu
from montblanc.src_types import source_var_types
from montblanc.solvers import MontblancTensorflowSolver

from . import load_tf_lib
from .cube_dim_transcoder import CubeDimensionTranscoder
from .staging_area_wrapper import create_staging_area_wrapper
from .sources import (SourceContext, DefaultsSourceProvider)
from .sinks import (SinkContext, NullSinkProvider)
from .start_context import StartContext
from .stop_context import StopContext
from .init_context import InitialisationContext

ONE_KB, ONE_MB, ONE_GB = 1024, 1024**2, 1024**3

QUEUE_SIZE = 10

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

        _setup_hypercube(cube, slvr_cfg)

        #=======================
        # Data Sources and Sinks
        #=======================

        # Get the defaults data source (default or test data)
        data_source = slvr_cfg['data_source']
        montblanc.log.info("Defaults Data Source '{}'".format(data_source))

        # Construct list of data sources and sinks
        # internal to the solver.
        # These will be overridden by source and sink
        # providers supplied by the user in the solve()
        # method
        default_prov = _create_defaults_source_provider(cube, data_source)
        self._source_providers = [default_prov]
        self._sink_providers = [NullSinkProvider()]

        #==================
        # Data Source Cache
        #==================

        class SourceCache(object):
            def __init__(self):
                self._cache = {}
                self._lock = threading.Lock()

            def __getitem__(self, key):
                with self._lock:
                    return self._cache[key]

            def __setitem__(self, key, value):
                with self._lock:
                    self._cache[key]=value

            def __delitem__(self, key):
                with self._lock:
                    del self._cache[key]

            def pop(self, key, default=None):
                with self._lock:
                    return self._cache.pop(key, default)


        self._source_cache = SourceCache()

        #==================
        # Memory Budgeting
        #==================

        # For deciding whether to rebudget
        self._previous_budget = 0
        self._previous_budget_dims = {}

        #================
        # Cube Transcoder
        #================
        self._iter_dims = ['ntime', 'nbl']
        self._transcoder = CubeDimensionTranscoder(self._iter_dims)

        #================================
        # Staging Area Data Source Configuration
        #================================

        dfs = { n: a for n, a in cube.arrays().iteritems()
            if not 'temporary' in a.tags }

        # Descriptors are not user-defined arrays
        # but a variable passed through describing a chunk of the
        # problem. Make it look as if it's an array
        if 'descriptor' in dfs:
            raise KeyError("'descriptor' is reserved, "
                "please use another array name.")

        dfs['descriptor'] = AttrDict(dtype=np.int32)

        #=========================
        # Tensorflow devices
        #=========================

        from tensorflow.python.client import device_lib
        devices = device_lib.list_local_devices()

        device_type = slvr_cfg['device_type'].upper()

        gpus = [d.name for d in devices if d.device_type == 'GPU']
        cpus = [d.name for d in devices if d.device_type == 'CPU']

        if device_type == 'GPU' and len(gpus) == 0:
            montblanc.log.warn("No GPUs are present, falling back to CPU.")
            device_type = 'CPU'

        use_cpus = device_type == 'CPU'
        montblanc.log.info("Using '{}' devices for compute".format(device_type))
        self._devices = cpus if use_cpus else gpus
        self._shards_per_device = spd = 2
        self._nr_of_shards = shards = len(self._devices)*spd
        # shard_id == d*spd + shard
        self._shard = lambda d, s: d*spd + s

        assert len(self._devices) > 0

        #=========================
        # Tensorflow Compute Graph
        #=========================

        # Create all tensorflow constructs within the compute graph
        with tf.Graph().as_default() as compute_graph:
            # Create our data feeding structure containing
            # input/output staging_areas and feed once variables
            self._tf_feed_data = _construct_tensorflow_feed_data(
                dfs, cube, self._iter_dims, shards)

            # Construct tensorflow expressions for each shard
            self._tf_expr = [_construct_tensorflow_expression(
                    slvr_cfg,
                    self._tf_feed_data, dev, self._shard(d,s))
                for d, dev in enumerate(self._devices)
                for s in range(self._shards_per_device)]

            # Initialisation operation
            init_op = tf.global_variables_initializer()
            # Now forbid modification of the graph
            compute_graph.finalize()

        #==========================================
        # Tensorflow Session
        #==========================================

        # Create the tensorflow session object
        # Use supplied target, if present
        tf_server_target = slvr_cfg.get('tf_server_target', '')

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

        self._descriptor_executor = tpe(1)
        self._feed_executors = [tpe(1) for i in range(shards)]
        self._compute_executors = [tpe(1) for i in range(shards)]
        self._consumer_executor = tpe(1)

        class InputsWaiting(object):
            """
            Keep track of the number of inputs waiting
            to be consumed on each shard
            """
            def __init__(self, shards):
                self._lock = threading.Lock()
                self._inputs_waiting = np.zeros(shape=(shards,), dtype=np.int32)

            def get(self):
                with self._lock:
                    return self._inputs_waiting

            def increment(self, shard):
                with self._lock:
                    self._inputs_waiting[shard] += 1

            def decrement(self, shard):
                with self._lock:
                    self._inputs_waiting[shard] -= 1

        self._inputs_waiting = InputsWaiting(shards)

        #======================
        # Tracing
        #======================

        class RunMetaData(object):
            def __init__(self):
                self._rm = []
                self._lock = threading.Lock()

            def clear(self):
                with self._lock:
                    self._rm = []

            def save(self, run_metadata):
                with self._lock:
                    self._rm.append(run_metadata)

            def write(self, tag=None):
                with self._lock:
                    if len(self._rm) == 0:
                        return

                    if tag is None:
                        tag='0'

                    metadata = tf.RunMetadata()
                    [metadata.MergeFrom(m) for m in self._rm]

                    tl = timeline.Timeline(metadata.step_stats)
                    trace_filename = 'compute_timeline_%d.json' % tag
                    with open(trace_filename, 'w') as f:
                        f.write(tl.generate_chrome_trace_format())
                        f.write('\n')

        #============================
        # Wrap tensorflow Session.run
        #============================

        self._should_trace = False
        self._run_metadata = RunMetaData()

        def _tfrunner(session, should_trace=False):
            """ Wrap the tensorflow Session.run method """
            trace_level = (tf.RunOptions.FULL_TRACE if should_trace
                                            else tf.RunOptions.NO_TRACE)
            options = tf.RunOptions(trace_level=trace_level)

            def _runner(*args, **kwargs):
                """ Pass options through """
                return session.run(*args, options=options, **kwargs)

            def _meta_runner(*args, **kwargs):
                """ Aggregate run metadata for each run """
                try:
                    run_metadata = tf.RunMetadata()
                    return session.run(*args, options=options,
                                              run_metadata=run_metadata,
                                            **kwargs)
                finally:
                    self._run_metadata.save(run_metadata)

            return _meta_runner if should_trace else _runner

        self._tfrun = _tfrunner(self._tf_session, self._should_trace)
        self._iterations = 0

    def _descriptor_feed(self):
        try:
            self._descriptor_feed_impl()
        except Exception as e:
            montblanc.log.exception("Descriptor Exception")
            raise

    def _descriptor_feed_impl(self):
        session = self._tf_session

        # Copy dimensions of the main cube
        cube = self.hypercube.copy()
        LSA = self._tf_feed_data.local

        # Get space of iteration
        iter_args = _iter_args(self._iter_dims, cube)
        descriptors_fed = 0

        # Iterate through the hypercube space
        for i, iter_cube in enumerate(cube.cube_iter(*iter_args)):
            descriptor = self._transcoder.encode(iter_cube.dimensions(copy=False))
            feed_dict = {LSA.descriptor.placeholders[0] : descriptor }
            montblanc.log.debug('Encoding {i} {d}'.format(i=i, d=descriptor))
            session.run(LSA.descriptor.put_op, feed_dict=feed_dict)
            descriptors_fed += 1

        montblanc.log.info("Done feeding {n} descriptors.".format(
            n=descriptors_fed))

        feed_dict = {LSA.descriptor.placeholders[0] : [-1] }
        session.run(LSA.descriptor.put_op, feed_dict=feed_dict)

    def _feed(self, cube, data_sources, data_sinks, global_iter_args):
        """ Feed stub """
        try:
            self._feed_impl(cube, data_sources, data_sinks, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_impl(self, cube, data_sources, data_sinks, global_iter_args):
        """ Implementation of staging_area feeding """
        session = self._tf_session
        FD = self._tf_feed_data
        LSA = FD.local

        # Get source strides out before the local sizes are modified during
        # the source loops below
        src_types = LSA.sources.keys()
        src_strides = [int(i) for i in cube.dim_extent_size(*src_types)]
        src_staging_areas = [[LSA.sources[t][s] for t in src_types]
            for s in range(self._nr_of_shards)]

        compute_feed_dict = { ph: cube.dim_global_size(n) for
            n, ph in FD.src_ph_vars.iteritems() }
        compute_feed_dict.update({ ph: getattr(cube, n) for
            n, ph in FD.property_ph_vars.iteritems() })

        chunks_fed = 0

        which_shard = itertools.cycle([self._shard(d,s)
            for s in range(self._shards_per_device)
            for d, dev in enumerate(self._devices)])

        while True:
            try:
                # Get the descriptor describing a portion of the RIME
                result = session.run(LSA.descriptor.get_op)
                descriptor = result['descriptor']
            except tf.errors.OutOfRangeError as e:
                montblanc.log.exception("Descriptor reading exception")

            # Quit if EOF
            if descriptor[0] == -1:
                break

            # Make it read-only so we can hash the contents
            descriptor.flags.writeable = False

            # Find indices of the emptiest staging_areas and, by implication
            # the shard with the least work assigned to it
            emptiest_staging_areas = np.argsort(self._inputs_waiting.get())
            shard = emptiest_staging_areas[0]
            shard = which_shard.next()

            feed_f = self._feed_executors[shard].submit(self._feed_actual,
                data_sources.copy(), cube.copy(),
                descriptor, shard,
                src_types, src_strides, src_staging_areas[shard],
                global_iter_args)

            compute_f = self._compute_executors[shard].submit(self._compute,
                compute_feed_dict, shard)

            consume_f = self._consumer_executor.submit(self._consume,
                data_sinks.copy(), cube.copy(), global_iter_args)

            self._inputs_waiting.increment(shard)

            yield (feed_f, compute_f, consume_f)

            chunks_fed += 1

        montblanc.log.info("Done feeding {n} chunks.".format(n=chunks_fed))

    def _feed_actual(self, *args):
        try:
            return self._feed_actual_impl(*args)
        except Exception as e:
            montblanc.log.exception("Feed Exception")
            raise

    def _feed_actual_impl(self, data_sources, cube,
            descriptor, shard,
            src_types, src_strides, src_staging_areas,
            global_iter_args):

        session = self._tf_session
        iq = self._tf_feed_data.local.feed_many[shard]

        # Decode the descriptor and update our cube dimensions
        dims = self._transcoder.decode(descriptor)
        cube.update_dimensions(dims)

        # Determine array shapes and data types for this
        # portion of the hypercube
        array_schemas = cube.arrays(reify=True)

        # Inject a data source and array schema for the
        # descriptor staging_area items.
        # These aren't full on arrays per se
        # but they need to work within the feeding framework
        array_schemas['descriptor'] = descriptor
        data_sources['descriptor'] = DataSource(
            lambda c: descriptor, np.int32, 'Internal')

        # Generate (name, placeholder, datasource, array schema)
        # for the arrays required by each staging_area
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
        self._source_cache[descriptor.data] = input_cache

        montblanc.log.info("Enqueueing chunk {d} on shard {sh}".format(
            d=descriptor, sh=shard))

        self._tfrun(iq.put_op, feed_dict=feed_dict)

        # For each source type, feed that source staging_area
        for src_type, staging_area, stride in zip(src_types, src_staging_areas, src_strides):
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
                # for the arrays required by each staging_area
                gen = [(a, ph, data_sources[a], array_schemas[a])
                    for ph, a in zip(staging_area.placeholders, staging_area.fed_arrays)]

                # Create a feed dictionary by calling the data source functors
                feed_dict = { ph: _get_data(ds, SourceContext(a, cube,
                        self.config(), global_iter_args + iter_args,
                        cube.array(a) if a in cube.arrays() else {},
                        ad.shape, ad.dtype))
                    for (a, ph, ds, ad) in gen }

                self._tfrun(staging_area.put_op, feed_dict=feed_dict)

    def _compute(self, feed_dict, shard):
        """ Call the tensorflow compute """

        try:
            descriptor, enq = self._tfrun(self._tf_expr[shard], feed_dict=feed_dict)
            self._inputs_waiting.decrement(shard)

        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise


    def _consume(self, data_sinks, cube, global_iter_args):
        """ Consume stub """
        try:
            return self._consume_impl(data_sinks, cube, global_iter_args)
        except Exception as e:
            montblanc.log.exception("Consumer Exception")
            raise e, None, sys.exc_info()[2]

    def _consume_impl(self, data_sinks, cube, global_iter_args):
        """ Consume """

        LSA = self._tf_feed_data.local
        output = self._tfrun(LSA.output.get_op)

        # Expect the descriptor in the first tuple position
        assert len(output) > 0
        assert LSA.output.fed_arrays[0] == 'descriptor'

        descriptor = output['descriptor']
        # Make it read-only so we can hash the contents
        descriptor.flags.writeable = False

        dims = self._transcoder.decode(descriptor)
        cube.update_dimensions(dims)

        # Obtain and remove input data from the source cache
        try:
            input_data = self._source_cache.pop(descriptor.data)
        except KeyError:
            raise ValueError("No input data cache available "
                "in source cache for descriptor {}!"
                    .format(descriptor))

        # For each array in our output, call the associated data sink
        gen = ((n, a) for n, a in output.iteritems() if not n == 'descriptor')

        for n, a in gen:
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

        src_provs_str = 'Source Providers ' + str([sp.name() for sp
                                                in source_providers])
        snk_provs_str = 'Sink Providers ' + str([sp.name() for sp
                                                in sink_providers])

        montblanc.log.info(src_provs_str)
        montblanc.log.info(snk_provs_str)

        # Allow providers to initialise themselves based on
        # the given configuration
        ctx = InitialisationContext(self.config())

        for p in itertools.chain(source_providers, sink_providers):
            p.init(ctx)

        # Apply any dimension updates from the source provider
        # to the hypercube, taking previous reductions into account
        bytes_required = _apply_source_provider_dim_updates(
            self.hypercube, source_providers,
            self._previous_budget_dims)

        # If we use more memory than previously,
        # perform another budgeting operation
        # to make sure everything fits
        if bytes_required > self._previous_budget:
            self._previous_budget_dims, self._previous_budget = (
                _budget(self.hypercube, self.config()))

        # Determine the global iteration arguments
        # e.g. [('ntime', 100), ('nbl', 20)]
        global_iter_args = _iter_args(self._iter_dims, self.hypercube)

        # Indicate solution started in providers
        ctx = StartContext(self.hypercube, self.config(), global_iter_args)

        for p in itertools.chain(source_providers, sink_providers):
            p.start(ctx)

        #===================================
        # Assign data to Feed Once variables
        #===================================

        # Copy the hypercube
        cube = self.hypercube.copy()
        array_schemas = cube.arrays(reify=True)

        # Construct data sources from those supplied by the
        # source providers, if they're associated with
        # input sources
        LSA = self._tf_feed_data.local
        input_sources = LSA.input_sources
        data_sources = {n: DataSource(f, cube.array(n).dtype, prov.name())
            for prov in source_providers
            for n, f in prov.sources().iteritems()
            if n in input_sources}

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
            in LSA.feed_once.iteritems() }

        self._run_metadata.clear()

        # Run the assign operations for each feed_once variable
        assign_ops = [fo.assign_op.op for fo in LSA.feed_once.itervalues()]
        self._tfrun(assign_ops, feed_dict=feed_dict)

        try:
            # Run the descriptor executor immediately
            params = self._descriptor_executor.submit(self._descriptor_feed)

            # Sets to track futures not yet completed
            feed_not_done = set()
            compute_not_done = set([params])
            consume_not_done = set()
            throttle_factor = self._nr_of_shards*QUEUE_SIZE

            # _feed_impl generates 3 futures
            # one for feeding data, one for computing with this data
            # and another for consuming it.
            # Iterate over these futures
            for feed, compute, consume in self._feed_impl(cube,
                data_sources, data_sinks, global_iter_args):

                feed_not_done.add(feed)
                compute_not_done.add(compute)
                consume_not_done.add(consume)

                # If there are many feed futures in flight,
                # perform throttling
                if len(feed_not_done) > throttle_factor*2:
                    # Wait for throttle_factor futures to complete
                    fit = cf.as_completed(feed_not_done)
                    feed_done = set(itertools.islice(fit, throttle_factor))
                    feed_not_done.difference_update(feed_done)

                    # Take an completed compute and consume
                    # futures immediately
                    compute_done, compute_not_done = cf.wait(
                        compute_not_done, timeout=0,
                        return_when=cf.FIRST_COMPLETED)
                    consume_done, consume_not_done = cf.wait(
                        consume_not_done, timeout=0,
                        return_when=cf.FIRST_COMPLETED)

                    # Get future results, mainly to fire exceptions
                    for i, f in enumerate(itertools.chain(feed_done,
                                        compute_done, consume_done)):
                        f.result()

                    not_done = sum(len(s) for s in (feed_not_done,
                        compute_not_done, consume_not_done))

                    montblanc.log.debug("Consumed {} futures. "
                        "{} remaining".format(i, not_done))

            # Request future results, mainly for exceptions
            for f in cf.as_completed(itertools.chain(feed_not_done,
                    compute_not_done, consume_not_done)):

                f.result()

        except (KeyboardInterrupt, SystemExit) as e:
            montblanc.log.exception('Solving interrupted')
            raise
        except Exception:
            montblanc.log.exception('Solving exception')
            raise
        else:
            if self._should_trace:
                self._run_metadata.write(self._iterations)

            self._iterations += 1
        finally:
            # Indicate solution stopped in providers
            ctx = StopContext(self.hypercube, self.config(), global_iter_args)
            for p in itertools.chain(source_providers, sink_providers):
                p.stop(ctx)

            montblanc.log.info('Solution Completed')


    def close(self):
        # Shutdown thread executors
        self._descriptor_executor.shutdown()
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


def _create_defaults_source_provider(cube, data_source):
    """
    Create a DefaultsSourceProvider object. This provides default
    data sources for each array defined on the hypercube. The data sources
    may either by obtained from the arrays 'default' data source
    or the 'test' data source.
    """
    from montblanc.impl.rime.tensorflow.sources import (
        find_sources, DEFAULT_ARGSPEC)
    from montblanc.impl.rime.tensorflow.sources import constant_cache

    # Obtain default data sources for each array,
    # Just take from defaults if test data isn't specified
    staging_area_data_source = ('default' if not data_source == 'test'
                                                      else data_source)

    cache = True

    default_prov = DefaultsSourceProvider(cache=cache)

    # Create data sources on the source provider from
    # the cube array data sources
    for n, a in cube.arrays().iteritems():
        # Unnecessary for temporary arrays
        if 'temporary' in a.tags:
            continue

        # Obtain the data source
        data_source = a.get(staging_area_data_source)

        # Array marked as constant, decorate the data source
        # with a constant caching decorator
        if cache is True and 'constant' in a.tags:
            data_source = constant_cache(data_source)

        method = types.MethodType(data_source, default_prov)
        setattr(default_prov, n, method)

    def _sources(self):
        """
        Override the sources method to also handle lambdas that look like
        lambda s, c: ..., as defined in the config module
        """

        try:
            return self._sources
        except AttributeError:
            self._sources = find_sources(self, [DEFAULT_ARGSPEC] + [['s', 'c']])

        return self._sources

    # Monkey patch the sources method
    default_prov.sources = types.MethodType(_sources, default_prov)

    return default_prov

def _construct_tensorflow_feed_data(dfs, cube, iter_dims,
    nr_of_input_staging_areas):

    FD = AttrDict()
    # https://github.com/bcj/AttrDict/issues/34
    FD._setattr('_sequence_type', list)
    # Reference local staging_areas
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
                    if 'input' in a.tags]

    src_data_sources, feed_many, feed_once = _partition(iter_dims,
                                                        input_arrays)

    #=====================================
    # Descriptor staging area
    #=====================================

    local.descriptor = create_staging_area_wrapper('descriptors',
        ['descriptor'], dfs)

    #===========================================
    # Staging area for multiply fed data sources
    #===========================================

    # Create the staging_area for holding the feed many input
    local.feed_many = [create_staging_area_wrapper('feed_many_%d' % i,
                ['descriptor'] + [a.name for a in feed_many], dfs)
            for i in range(nr_of_input_staging_areas)]

    #=================================================
    # Staging areas for each radio source data sources
    #=================================================

    # Create the source array staging areas
    local.sources = { src_nr_var: [
            create_staging_area_wrapper('%s_%d' % (src_type, i),
            [a.name for a in src_data_sources[src_nr_var]], dfs)
            for i in range(nr_of_input_staging_areas)]

        for src_type, src_nr_var in source_var_types().iteritems()
    }

    #======================================
    # The single output staging_area
    #======================================

    local.output = create_staging_area_wrapper('output',
        ['descriptor', 'model_vis', 'chi_squared'], dfs)

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

    # Data sources from input staging_areas
    src_sa = [q for sq in local.sources.values() for q in sq]
    all_staging_areas = local.feed_many + src_sa
    input_sources = { a for q in all_staging_areas
                        for a in q.fed_arrays}
    # Data sources from feed once variables
    input_sources.update(local.feed_once.keys())

    local.input_sources = input_sources

    return FD

def _construct_tensorflow_expression(slvr_cfg, feed_data, device, shard):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)
    src_count = zero
    src_ph_vars = feed_data.src_ph_vars

    LSA = feed_data.local

    polarisation_type = slvr_cfg['polarisation_type']

    # Pull RIME inputs out of the feed staging_area
    # of the relevant shard, adding the feed once
    # inputs to the dictionary
    D = LSA.feed_many[shard].get_to_attrdict()
    D.update({k: fo.var for k, fo in LSA.feed_once.iteritems()})

    with tf.device(device):
        # Infer chunk dimensions
        model_vis_shape = tf.shape(D.model_vis)
        ntime, nbl, nchan, npol = [model_vis_shape[i] for i in range(4)]

        # Infer float and complex type
        FT, CT = D.uvw.dtype, D.model_vis.dtype

        # Compute sine and cosine of parallactic angles
        pa_sin, pa_cos = rime.parallactic_angle_sin_cos(D.parallactic_angles)
        # Compute feed rotation
        feed_rotation = rime.feed_rotation(pa_sin, pa_cos, CT=CT,
                                           feed_type=polarisation_type)

    def antenna_jones(lm, stokes, alpha, ref_freq):
        """
        Compute the jones terms for each antenna.

        lm, stokes and alpha are the source variables.
        """

        # Compute the complex phase
        cplx_phase = rime.phase(lm, D.uvw, D.frequency, CT=CT)

        # Check for nans/infs in the complex phase
        phase_msg = ("Check that '1 - l**2  - m**2 >= 0' holds "
                    "for all your lm coordinates. This is required "
                    "for 'n = sqrt(1 - l**2 - m**2) - 1' "
                    "to be finite.")

        phase_real = tf.check_numerics(tf.real(cplx_phase), phase_msg)
        phase_imag = tf.check_numerics(tf.imag(cplx_phase), phase_msg)

        # Compute the square root of the brightness matrix
        # (as well as the sign)
        bsqrt, sgn_brightness = rime.b_sqrt(stokes, alpha,
            D.frequency, ref_freq, CT=CT,
            polarisation_type=polarisation_type)

        # Check for nans/infs in the bsqrt
        bsqrt_msg = ("Check that your stokes parameters "
                    "satisfy I**2 >= Q**2 + U**2 + V**2. "
                    "Montblanc performs a cholesky decomposition "
                    "of the brightness matrix and the above must "
                    "hold for this to produce valid values.")

        bsqrt_real = tf.check_numerics(tf.real(bsqrt), bsqrt_msg)
        bsqrt_imag = tf.check_numerics(tf.imag(bsqrt), bsqrt_msg)

        # Compute the direction dependent effects from the beam
        ejones = rime.e_beam(lm, D.frequency,
            D.pointing_errors, D.antenna_scaling,
            pa_sin, pa_cos,
            D.beam_extents, D.beam_freq_map, D.ebeam)

        deps = [phase_real, phase_imag, bsqrt_real, bsqrt_imag]
        deps = [] # Do nothing for now

        # Combine the brightness square root, complex phase,
        # feed rotation and beam dde's
        with tf.control_dependencies(deps):
            antenna_jones = rime.create_antenna_jones(bsqrt, cplx_phase,
                                                    feed_rotation, ejones, FT=FT)
            return antenna_jones, sgn_brightness

    # While loop condition for each point source type
    def point_cond(coherencies, npsrc, src_count):
        return tf.less(npsrc, src_ph_vars.npsrc)

    def gaussian_cond(coherencies, ngsrc, src_count):
        return tf.less(ngsrc, src_ph_vars.ngsrc)

    def sersic_cond(coherencies, nssrc, src_count):
        return tf.less(nssrc, src_ph_vars.nssrc)

    # While loop bodies
    def point_body(coherencies, npsrc, src_count):
        """ Accumulate visiblities for point source batch """
        S = LSA.sources['npsrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.point_lm)[0]
        src_count += nsrc
        npsrc +=  nsrc

        ant_jones, sgn_brightness = antenna_jones(S.point_lm,
            S.point_stokes, S.point_alpha, S.point_ref_freq)
        shape = tf.ones(shape=[nsrc,ntime,nbl,nchan], dtype=FT)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, npsrc, src_count

    def gaussian_body(coherencies, ngsrc, src_count):
        """ Accumulate coherencies for gaussian source batch """
        S = LSA.sources['ngsrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.gaussian_lm)[0]
        src_count += nsrc
        ngsrc += nsrc

        ant_jones, sgn_brightness = antenna_jones(S.gaussian_lm,
            S.gaussian_stokes, S.gaussian_alpha, S.gaussian_ref_freq)
        gauss_shape = rime.gauss_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, S.gaussian_shape)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            gauss_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, ngsrc, src_count

    def sersic_body(coherencies, nssrc, src_count):
        """ Accumulate coherencies for sersic source batch """
        S = LSA.sources['nssrc'][shard].get_to_attrdict()

        # Maintain source counts
        nsrc = tf.shape(S.sersic_lm)[0]
        src_count += nsrc
        nssrc += nsrc

        ant_jones, sgn_brightness = antenna_jones(S.sersic_lm,
            S.sersic_stokes, S.sersic_alpha, S.sersic_ref_freq)
        sersic_shape = rime.sersic_shape(D.uvw, D.antenna1, D.antenna2,
            D.frequency, S.sersic_shape)
        coherencies = rime.sum_coherencies(D.antenna1, D.antenna2,
            sersic_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, nssrc, src_count

    with tf.device(device):
        base_coherencies = tf.zeros(shape=[ntime,nbl,nchan,npol], dtype=CT)

        # Evaluate point sources
        summed_coherencies, npsrc, src_count = tf.while_loop(
            point_cond, point_body,
            [base_coherencies, zero, src_count])

        # Evaluate gaussians
        summed_coherencies, ngsrc, src_count = tf.while_loop(
            gaussian_cond, gaussian_body,
            [summed_coherencies, zero, src_count])

        # Evaluate sersics
        summed_coherencies, nssrc, src_count = tf.while_loop(
            sersic_cond, sersic_body,
            [summed_coherencies, zero, src_count])

        # Post process visibilities to produce model visibilites and chi squared
        model_vis, chi_squared = rime.post_process_visibilities(
            D.antenna1, D.antenna2, D.direction_independent_effects, D.flag,
            D.weight, D.model_vis, summed_coherencies, D.observed_vis)

    # Create enstaging_area operation
    put_op = LSA.output.put_from_list([D.descriptor, model_vis, chi_squared])

    # Return descriptor and enstaging_area operation
    return D.descriptor, put_op

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

def _budget(cube, slvr_cfg):
    # Figure out a viable dimension configuration
    # given the total problem size
    mem_budget = slvr_cfg.get('mem_budget', 2*ONE_GB)
    bytes_required = cube.bytes_required()

    src_dims = mbu.source_nr_vars() + ['nsrc']
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
        sbs = slvr_cfg['source_batch_size']
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

    return applied_reductions, bytes_required

DimensionUpdate = attr.make_class("DimensionUpdate",
    ['size', 'prov'], slots=True, frozen=True)

def _apply_source_provider_dim_updates(cube, source_providers, budget_dims):
    """
    Given a list of source_providers, apply the list of
    suggested dimension updates given in provider.updated_dimensions()
    to the supplied hypercube.

    Dimension global_sizes are always updated with the supplied sizes and
    lower_extent is always set to 0. upper_extent is set to any reductions
    (current upper_extents) existing in budget_dims, otherwise it is set
    to global_size.

    """
    # Create a mapping between a dimension and a
    # list of (global_size, provider_name) tuples
    update_map = collections.defaultdict(list)

    for prov in source_providers:
        for dim_tuple in prov.updated_dimensions():
            name, size = dim_tuple

            # Don't accept any updates on the nsrc dimension
            # This is managed internally
            if name == 'nsrc':
                continue

            dim_update = DimensionUpdate(size, prov.name())
            update_map[name].append(dim_update)

    # No dimensions were updated, quit early
    if len(update_map) == 0:
        return cube.bytes_required()

    # Ensure that the global sizes we receive
    # for each dimension are unique. Tell the user
    # when conflicts occur
    update_list = []

    for name, updates in update_map.iteritems():
        if not all(updates[0].size == du.size for du in updates[1:]):
            raise ValueError("Received conflicting "
                "global size updates '{u}'"
                " for dimension '{n}'.".format(n=name, u=updates))

        update_list.append((name, updates[0].size))

    montblanc.log.info("Updating dimensions {} from "
                        "source providers.".format(str(update_list)))

    # Now update our dimensions
    for name, global_size in update_list:
        # Defer to existing any existing budgeted extent sizes
        # Otherwise take the global_size
        extent_size = budget_dims.get(name, global_size)
        # Clamp extent size to global size
        if extent_size > global_size:
            extent_size = global_size

        # Update the dimension
        cube.update_dimension(name,
            global_size=global_size,
            lower_extent=0,
            upper_extent=extent_size)

    # Handle global number of sources differently
    # It's equal to the number of
    # point's, gaussian's, sersic's combined
    nsrc = sum(cube.dim_global_size(*mbu.source_nr_vars()))

    # Extent size will be equal to whatever source type
    # we're currently iterating over. So just take
    # the maximum extent size given the sources
    es = max(cube.dim_extent_size(*mbu.source_nr_vars()))

    cube.update_dimension('nsrc',
        global_size=nsrc,
        lower_extent=0,
        upper_extent=es)

    # Return our cube size
    return cube.bytes_required()

def _setup_hypercube(cube, slvr_cfg):
    """ Sets up the hypercube given a solver configuration """
    mbu.register_default_dimensions(cube, slvr_cfg)

    # Configure the dimensions of the beam cube
    cube.register_dimension('beam_lw', 2,
                            description='E Beam cube l width')

    cube.register_dimension('beam_mh', 2,
                            description='E Beam cube m height')

    cube.register_dimension('beam_nud', 2,
                            description='E Beam cube nu depth')

    # =========================================
    # Register hypercube Arrays and Properties
    # =========================================

    from montblanc.impl.rime.tensorflow.config import (A, P)

    def _massage_dtypes(A, T):
        def _massage_dtype_in_dict(D):
            new_dict = D.copy()
            new_dict['dtype'] = mbu.dtype_from_str(D['dtype'], T)
            return new_dict

        return [_massage_dtype_in_dict(D) for D in A]

    dtype = slvr_cfg['dtype']
    is_f32 = dtype == 'float'

    T = {
        'ft' : np.float32 if is_f32 else np.float64,
        'ct' : np.complex64 if is_f32 else np.complex128,
        'int' : int,
    }

    cube.register_properties(_massage_dtypes(P, T))
    cube.register_arrays(_massage_dtypes(A, T))

def _partition(iter_dims, data_sources):
    """
    Partition data sources into

    1. Dictionary of data sources associated with radio sources.
    2. List of data sources to feed multiple times.
    3. List of data sources to feed once.
    """

    src_nr_vars = set(source_var_types().values())
    iter_dims = set(iter_dims)

    src_data_sources = collections.defaultdict(list)
    feed_many = []
    feed_once = []

    for ds in data_sources:
        # Is this data source associated with
        # a radio source (point, gaussian, etc.?)
        src_int = src_nr_vars.intersection(ds.shape)

        if len(src_int) > 1:
            raise ValueError("Data source '{}' contains multiple "
                            "source types '{}'".format(ds.name, src_int))
        elif len(src_int) == 1:
            # Yep, record appropriately and iterate
            src_data_sources[src_int.pop()].append(ds)
            continue

        # Are we feeding this data source multiple times
        # (Does it possess dimensions on which we iterate?)
        if len(iter_dims.intersection(ds.shape)) > 0:
            feed_many.append(ds)
            continue

        # Assume this is a data source that we only feed once
        feed_once.append(ds)

    return src_data_sources, feed_many, feed_once
