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
FEED_ONCE_KEY = 0

rime = load_tf_lib()

DataSource = attr.make_class("DataSource", ['source', 'dtype', 'name'],
    slots=True, frozen=True)
DataSink = attr.make_class("DataSink", ['sink', 'name'],
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
        self._devices = cpus if device_type == 'CPU' else gpus

        assert len(self._devices) > 0

        self._ndevices = ndevices = len(self._devices)

        #=========================
        # Tensorflow Compute Graph
        #=========================

        # Create all tensorflow constructs within the compute graph
        with tf.Graph().as_default() as compute_graph:
            # Create our data feeding structure containing
            # input/output staging_areas and feed once variables
            self._tf_feed_data = _construct_tensorflow_staging_areas(
                cube, self._iter_dims, self._devices)

            # Construct tensorflow expressions for each device
            self._tf_expr = [_construct_tensorflow_expression(
                                self._tf_feed_data, slvr_cfg, dev, d)
                for d, dev in enumerate(self._devices)]

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

        session = tf.Session(tf_server_target, graph=compute_graph,
                                                config=session_config)

        from tensorflow.python import debug as tf_debug

        self._tf_session = session
        #self._tf_session = tf_debug.LocalCLIDebugWrapperSession(session)

        self._tf_session.run(init_op)

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


    def _compute(self, feed_dict, dev_id):
        """ Call the tensorflow compute """

        try:
            expr = self._tf_expr[dev_id]
            self._tfrun([expr.stage_feed_many,
                         expr.stage_output,
                         expr.stage_cpu_output],
                            feed_dict=feed_dict)
        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise

    def close(self):
        # Shutdown the tensorflow session
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

def _construct_tensorflow_staging_areas(cube, iter_dims, devices):

    cpu_dev = tf.DeviceSpec(device_type='CPU')

    FD = AttrDict()
    # https://github.com/bcj/AttrDict/issues/34
    FD._setattr('_sequence_type', list)

    # Reference local staging_areas on the CPU
    FD.local_cpu = local_cpu = AttrDict()
    local_cpu._setattr('_sequence_type', list)

    # Reference local staging areas on compute device (GPUs)
    FD.local_compute = local_compute = AttrDict()
    local_compute._setattr('_sequence_type', list)

    # Create placeholder variables for source counts
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
    input_arrays = { n: a for n, a in cube.arrays().iteritems()
                                        if 'input' in a.tags }

    src_data_sources, feed_many, feed_once = _partition(iter_dims,
                                                        input_arrays)

    #======================================
    # Staging area for fed once data sources
    #======================================

    with tf.device(cpu_dev):
        local_cpu.feed_once = create_staging_area_wrapper('feed_once_cpu',
            [a.name for a in feed_once], input_arrays, ordered=True)

    # Create the staging_areas on the compute devices
    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            saw = create_staging_area_wrapper(
                'feed_once_compute_%d' % i,
                [a.name for a in feed_once],
                input_arrays, ordered=True)
            staging_areas.append(saw)

    local_compute.feed_once = staging_areas

    #===========================================
    # Staging area for multiply fed data sources
    #===========================================

    # Create the staging_area for holding the feed many input
    with tf.device(cpu_dev):
        local_cpu.feed_many = create_staging_area_wrapper(
                    'feed_many_cpu',
                    [a.name for a in feed_many],
                    input_arrays, ordered=True)

    # Create the staging_areas on the compute devices
    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            saw = create_staging_area_wrapper(
                'feed_many_compute_%d' % i,
                [a.name for a in feed_many],
                input_arrays, ordered=True)
            staging_areas.append(saw)

    local_compute.feed_many = staging_areas

    #=================================================
    # Staging areas for each radio source data sources
    #=================================================

    # Create the source array staging areas
    with tf.device(cpu_dev):
        local_cpu.sources = { src_type: create_staging_area_wrapper(
                '%s_cpu' % src_type,
                [a.name for a in src_data_sources[src_type]],
                input_arrays, ordered=True)

            for src_type, src_nr_var in source_var_types().iteritems()
        }

    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            # Create the source array staging areas
            saws = { src_type: create_staging_area_wrapper(
                '%s_compute_%d' % (src_type, i),
                [a.name for a in src_data_sources[src_type]],
                input_arrays, ordered=True)

                 for src_type, src_nr_var in source_var_types().iteritems()
             }
            staging_areas.append(saws)

    local_compute.sources = staging_areas

    #======================================
    # The single output staging_area
    #======================================

    output_arrays = { n: a for n, a in cube.arrays().iteritems()
                                            if 'output' in a.tags }

    for i, dev in enumerate(devices):
        with tf.device(dev):
            local_compute.output = create_staging_area_wrapper(
                'output', output_arrays.keys(),
                output_arrays, ordered=True)

    with tf.device(cpu_dev):
        local_cpu.output = create_staging_area_wrapper(
            'output',  output_arrays.keys(),
            output_arrays, ordered=True)

    #=======================================================
    # Construct the list of data sources that need feeding
    #=======================================================

    # Data sources from input staging_areas
    src_sa = local_cpu.sources.values()
    all_staging_areas = [local_cpu.feed_many] + [local_cpu.feed_once] + src_sa
    input_sources = { a for q in all_staging_areas
                        for a in q.fed_arrays}
    # Data sources from feed once variables
    input_sources.update(local_cpu.feed_once.fed_arrays)

    local_cpu.all_staging_areas = all_staging_areas
    local_cpu.input_sources = input_sources

    src_sa = [sa for devsa in local_compute.sources for sa in devsa.values()]
    all_staging_areas = local_compute.feed_many + local_compute.feed_once + src_sa
    local_compute.all_staging_areas = all_staging_areas

    return FD

def _construct_tensorflow_expression(feed_data, slvr_cfg, device, dev_id):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)
    src_count = zero
    src_ph_vars = feed_data.src_ph_vars

    local_cpu = feed_data.local_cpu
    local_compute = feed_data.local_compute

    polarisation_type = slvr_cfg['polarisation_type']

    # Create ops for copying from the CPU to the compute staging area
    key, data = local_cpu.feed_once.get(FEED_ONCE_KEY)
    stage_feed_once = local_compute.feed_once[dev_id].put(key, data)

    key, data = local_cpu.feed_many.get()
    stage_feed_many = local_compute.feed_many[dev_id].put(key, data)

    # Pull RIME inputs out of the feed many staging_area
    # for the relevant device, adding the feed once
    # inputs to the dictionary
    key, D = local_compute.feed_many[dev_id].get_to_attrdict(
                                                  name="compute_feed_many_get")
    D.update(local_compute.feed_once[dev_id].peek(FEED_ONCE_KEY,
                                                  name="compute_feed_once_peek"))

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
        key, S = local_cpu.sources['point'].get_to_attrdict()

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
        key, S = local_cpu.sources['gaussian'].get_to_attrdict()

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
        key, S = local_cpu.sources['sersic'].get_to_attrdict()

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

        # Post process visibilities to produce model visibilities and chi squared
        model_vis, chi_squared = rime.post_process_visibilities(
            D.antenna1, D.antenna2, D.direction_independent_effects, D.flag,
            D.weight, D.model_vis, summed_coherencies, D.observed_vis)

    # Create staging_area put operation
    stage_output = local_compute.output.put(key,
        {'model_vis': model_vis,'chi_squared': chi_squared})
        # Stage output in the compute output staging area
<<<<<<< 361a74f3647b4aee84478e85b0003320f32e7c60
        stage_output = local_compute.output.put(key,
=======
        stage_output = local_compute.output.put(key, {'model_vis': model_vis,
                                                'chi_squared': chi_squared})
>>>>>>> Add chi-squared to the output staging area

    # Create ops for shifting output from compute staging area
    # to CPU staging area
    out_key, out_data = local_compute.output.get(key)
    stage_cpu_output = local_cpu.output.put(out_key, out_data)

    ComputeNodes = attr.make_class("ComputeNodes", ["stage_feed_many",
                                                    "stage_feed_once",
                                                    "stage_output",
                                                    "stage_cpu_output"])

    # Return Compute operations
    return ComputeNodes(stage_feed_many,
                        stage_feed_once,
                        stage_output,
                        stage_cpu_output)

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

    # Map dimension to source types
    src_dims_to_types = { v: k for k, v in source_var_types().items() }

    src_dims = set(src_dims_to_types.keys())
    iter_dims = set(iter_dims)

    src_data_sources = collections.defaultdict(list)
    feed_many = []
    feed_once = []

    for n, ds in data_sources.iteritems():
        # Is this data source associated with
        # a radio source (point, gaussian, etc.?)
        src_int = src_dims.intersection(ds.shape)

        if len(src_int) > 1:
            raise ValueError("Data source '{}' contains multiple "
                            "source types '{}'".format(n, src_int))
        elif len(src_int) == 1:
            # Yep, record appropriately and iterate
            src_type = src_dims_to_types[src_int.pop()]
            src_data_sources[src_type].append(ds)
            continue

        # Are we feeding this data source multiple times
        # (Does it possess dimensions on which we iterate?)
        if len(iter_dims.intersection(ds.shape)) > 0:
            feed_many.append(ds)
            continue

        # Assume this is a data source that we only feed once
        feed_once.append(ds)

    return src_data_sources, feed_many, feed_once
