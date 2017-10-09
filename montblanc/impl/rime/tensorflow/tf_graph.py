import collections

import attr
from attrdict import AttrDict
import numpy as np
import six
import tensorflow as tf

from montblanc.src_types import source_var_types

from montblanc.impl.rime.tensorflow.staging_area_wrapper import create_staging_area_wrapper
from montblanc.impl.rime.tensorflow import load_tf_lib

rime = load_tf_lib()


def _partition(iter_dims, data_sources):
    """
    Partition data sources into

    1. Dictionary of dictionaries of data sources
       associated with radio sources.
    2. Dictionary of data sources to feed multiple times.
    3. Dictionary of data sources to feed once.
    """

    src_dims = set(source_var_types().keys())
    iter_dims = set(iter_dims)

    src_data_sources = collections.defaultdict(dict)
    feed_many = {}
    feed_once = {}

    for n, ds in six.iteritems(data_sources):
        # Is this data source associated with
        # a radio source (point, gaussian, etc.?)
        src_int = src_dims.intersection(ds["dims"])

        if len(src_int) > 1:
            raise ValueError("Data source '{}' contains multiple "
                            "source types '{}'".format(n, src_int))
        elif len(src_int) == 1:
            # Yep, record appropriately and iterate
            src_data_sources[src_int.pop()][n] = ds
            continue

        # Are we feeding this data source multiple times
        # (Does it possess dimensions on which we iterate?)
        if len(iter_dims.intersection(ds["dims"])) > 0:
            feed_many[n] = ds
            continue

        # Assume this is a data source that we only feed once
        feed_once[n] = ds

    return src_data_sources, feed_many, feed_once

def _construct_tensorflow_staging_areas(in_schema, out_schema,
                                            iter_dims, devices):

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

    #========================================================
    # Determine which arrays need feeding once/multiple times
    #========================================================

    src_data_sources, feed_many, feed_once = _partition(iter_dims,
                                                        in_schema)

    #=======================================
    # Staging area for internal data sources
    #=======================================

    internal_schema = { "%s_keys" % st: { "dtype" : np.int64 }
                                for st in src_data_sources.keys() }

    with tf.device(cpu_dev):
        local_cpu.feed_internal = create_staging_area_wrapper('internal',
            internal_schema.keys(),
            internal_schema, ordered=True)

    #======================================
    # Staging area for fed once data sources
    #======================================

    with tf.device(cpu_dev):
        local_cpu.feed_once = create_staging_area_wrapper('feed_once_cpu',
            feed_once.keys(), in_schema, ordered=True)

    # Create the staging_areas on the compute devices
    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            saw = create_staging_area_wrapper(
                'feed_once_compute_%d' % i,
                feed_once.keys(),
                in_schema, ordered=True)
            staging_areas.append(saw)

    local_compute.feed_once = staging_areas

    #===========================================
    # Staging area for multiply fed data sources
    #===========================================

    # Create the staging_area for holding the feed many input
    with tf.device(cpu_dev):
        local_cpu.feed_many = create_staging_area_wrapper(
                    'feed_many_cpu',
                    feed_many.keys(),
                    in_schema, ordered=True)

    # Create the staging_areas on the compute devices
    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            saw = create_staging_area_wrapper(
                'feed_many_compute_%d' % i,
                feed_many.keys(),
                in_schema, ordered=True)
            staging_areas.append(saw)

    local_compute.feed_many = staging_areas

    #=================================================
    # Staging areas for each radio source data sources
    #=================================================

    # Create the source array staging areas
    with tf.device(cpu_dev):
        local_cpu.sources = { src_type: create_staging_area_wrapper(
                '%s_cpu' % src_type,
                src_data_sources[src_type].keys(),
                in_schema, ordered=True)

            for src_type in source_var_types().keys()
        }

    staging_areas = []

    for i, dev in enumerate(devices):
        with tf.device(dev):
            # Create the source array staging areas
            saws = { src_type: create_staging_area_wrapper(
                    '%s_compute_%d' % (src_type, i),
                    src_data_sources[src_type].keys(),
                    in_schema, ordered=True)

                for src_type in source_var_types().keys()
             }
            staging_areas.append(saws)

    local_compute.sources = staging_areas

    #======================================
    # The single output staging_area
    #======================================

    for i, dev in enumerate(devices):
        with tf.device(dev):
            local_compute.output = create_staging_area_wrapper(
                'output', out_schema.keys(),
                out_schema, ordered=True)

    with tf.device(cpu_dev):
        local_cpu.output = create_staging_area_wrapper(
            'output',  out_schema.keys(),
            out_schema, ordered=True)

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

    local_cpu.feed_once_key = tf.placeholder(tf.int64, name="feed_once_key")
    local_cpu.feed_many_key = tf.placeholder(tf.int64, name="feed_many_key")

    return FD

def _construct_tensorflow_expression(feed_data, slvr_cfg, device, dev_id):
    """ Constructs a tensorflow expression for computing the RIME """
    zero = tf.constant(0)

    local_cpu = feed_data.local_cpu
    local_compute = feed_data.local_compute

    polarisation_type = slvr_cfg['polarisation_type']

    # Create ops for copying from the CPU to compute staging areas

    # Feed Once Staging Area
    data = local_cpu.feed_once.peek(local_cpu.feed_once_key,
                                    name="cpu_feed_once_peek")
    stage_feed_once = local_compute.feed_once[dev_id].put(
                                    local_cpu.feed_once_key, data,
                                    name="compute_feed_once_put")

    # Feed Many Staging Area
    feed_many_key, data = local_cpu.feed_many.get(local_cpu.feed_many_key,
                                        name="cpu_feed_many_get")
    stage_feed_many = local_compute.feed_many[dev_id].put(feed_many_key, data,
                                                  name="compute_feed_many_put")

    # Pull RIME inputs out of the feed many staging_area
    # for the relevant device, adding the feed once
    # inputs to the dictionary
    feed_many_key, D = local_compute.feed_many[dev_id].get_to_attrdict(local_cpu.feed_many_key,
                                                  name="compute_feed_many_get")
    D.update(local_compute.feed_once[dev_id].peek(local_cpu.feed_once_key,
                                                  name="compute_feed_once_peek"))

    # Get internal data for this computation
    _, I = local_cpu.feed_internal.get_to_attrdict(local_cpu.feed_many_key,
                                                name="compute_feed_internal_key")

    stage_source_loops = []

    for src_type in source_var_types().keys():
        keys = getattr(I, "%s_keys" % src_type)

        # How many chunks should be fed?
        nsrc_chunks = tf.cast(tf.shape(keys)[0], tf.int64)

        def cond(chunk):
            return tf.less(chunk, nsrc_chunks)

        def body(chunk):
            key, data = local_cpu.sources[src_type].get(keys[chunk],
                                        name="cpu_%s_get" % src_type)

            feed_src_chunk = local_compute.sources[dev_id][src_type].put(key, data,
                                        name="compute_%s_put" % src_type)

            with tf.control_dependencies([feed_src_chunk]):
                return [chunk + 1]

        loop = tf.while_loop(cond, body, [tf.constant(0,dtype=tf.int64)])
        stage_source_loops.append(loop)

    stage_source_data = tf.group(*stage_source_loops)

    # Infer chunk dimensions
    with tf.device(device):
        # Infer chunk dimensions
        model_vis_shape = tf.shape(D.data)
        nrow, nchan, npol = [model_vis_shape[i] for i in range(3)]

        # Infer float and complex type
        FT, CT = D.antenna_uvw.dtype, D.data.dtype

        # Compute sine and cosine of parallactic angles
        pa_sin, pa_cos = rime.parallactic_angle_sin_cos(D.parallactic_angles)
        # Compute feed rotation
        feed_rotation = rime.feed_rotation(pa_sin, pa_cos, CT=CT,
                                           feed_type=polarisation_type)

    def antenna_jones(lm, stokes, alpha, ref_freq):
        """
        Compute the jones terms for each antenna.

        `lm`, `stokes`, `alpha` and `ref_freq` are the source variables.
        """

        # Compute the complex phase
        cplx_phase = rime.phase(lm, D.antenna_uvw, D.frequency, CT=CT)

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

        ejones_msg = ("Invalid beam values")

        ejones_real = tf.check_numerics(tf.real(ejones), ejones_msg)
        ejones_imag = tf.check_numerics(tf.imag(ejones), ejones_msg)

        deps = [phase_real, phase_imag, bsqrt_real, bsqrt_imag, ejones_real, ejones_imag]
        deps = [] # Do nothing for now

        # Combine the brightness square root, complex phase,
        # feed rotation and beam dde's
        with tf.control_dependencies(deps):
            antenna_jones = rime.create_antenna_jones(bsqrt, cplx_phase,
                                                    feed_rotation, ejones, FT=FT)
            return antenna_jones, sgn_brightness

    npoint_chunks = tf.shape(I.point_keys)[0]
    ngaussian_chunks = tf.shape(I.gaussian_keys)[0]
    nsersic_chunks = tf.shape(I.sersic_keys)[0]

    # While loop condition for each point source type
    def point_cond(coherencies, chunk):
        return tf.less(chunk, npoint_chunks)

    def gaussian_cond(coherencies, chunk):
        return tf.less(chunk, ngaussian_chunks)

    def sersic_cond(coherencies, chunk):
        return tf.less(chunk, nsersic_chunks)

    # While loop bodies
    def point_body(coherencies, chunk):
        """ Accumulate visiblities for point source batch """
        point_sources = local_compute.sources[dev_id]['point']
        _, S = point_sources.get_to_attrdict(I.point_keys[chunk])

        # Get source count for this chunk
        nsrc = tf.shape(S.point_lm)[0]

        ant_jones, sgn_brightness = antenna_jones(S.point_lm,
            S.point_stokes, S.point_alpha, S.point_ref_freq)
        shape = tf.ones(shape=[nsrc,nrow,nchan], dtype=FT)
        coherencies = rime.sum_coherencies(D.time_index,
            D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, chunk + 1

    def gaussian_body(coherencies, chunk):
        """ Accumulate coherencies for gaussian source batch """
        gaussian_sources = local_compute.sources[dev_id]['gaussian']
        _, S = gaussian_sources.get_to_attrdict(I.gaussian_keys[chunk])

        ant_jones, sgn_brightness = antenna_jones(S.gaussian_lm,
            S.gaussian_stokes, S.gaussian_alpha, S.gaussian_ref_freq)
        gauss_shape = rime.gauss_shape(D.time_index, D.antenna_uvw,
            D.antenna1, D.antenna2,
            D.frequency, S.gaussian_shape_params)
        coherencies = rime.sum_coherencies(D.time_index,
            D.antenna1, D.antenna2,
            gauss_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, chunk + 1

    def sersic_body(coherencies, chunk):
        """ Accumulate coherencies for sersic source batch """
        sersic_sources = local_compute.sources[dev_id]['sersic']
        _, S = sersic_sources.get_to_attrdict(I.sersic_keys[chunk])

        ant_jones, sgn_brightness = antenna_jones(S.sersic_lm,
            S.sersic_stokes, S.sersic_alpha, S.sersic_ref_freq)
        sersic_shape = rime.sersic_shape(D.time_index, D.antenna_uvw,
            D.antenna1, D.antenna2,
            D.frequency, S.sersic_shape_params)
        coherencies = rime.sum_coherencies(D.time_index,
            D.antenna1, D.antenna2,
            sersic_shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, chunk + 1

    with tf.device(device):
        base_coherencies = tf.zeros(shape=[nrow,nchan,npol], dtype=CT)

        # Evaluate point sources
        summed_coherencies, point_chunks = tf.while_loop(point_cond,
                                                point_body,
                                                [base_coherencies, zero])

        # Evaluate gaussians
        summed_coherencies, gaussian_chunks = tf.while_loop(gaussian_cond,
                                                gaussian_body,
                                                [summed_coherencies, zero])

        # Evaluate sersics
        summed_coherencies, sersic_chunks = tf.while_loop(sersic_cond,
                                                sersic_body,
                                                [summed_coherencies, zero])

        # Post process visibilities to produce model visibilities and chi squared
        model_vis, chi_squared = rime.post_process_visibilities(
            D.time_index, D.antenna1, D.antenna2,
            D.direction_independent_effects, D.flag,
            D.weight, D.data, summed_coherencies, D.data)

        # Stage output in the compute output staging area
        stage_output = local_compute.output.put(feed_many_key,
                            { 'model_vis': model_vis,
                             'chi_squared': chi_squared })

    # Create ops for shifting output from compute staging area
    # to CPU staging area
    out_key, out_data = local_compute.output.get(feed_many_key)
    stage_cpu_output = local_cpu.output.put(out_key, out_data)

    ComputeNodes = attr.make_class("ComputeNodes", ["stage_feed_many",
                                                    "stage_feed_once",
                                                    "stage_source_data",
                                                    "stage_output",
                                                    "stage_cpu_output"])

    # Return Compute operations
    return ComputeNodes(stage_feed_many,
                        stage_feed_once,
                        stage_source_data,
                        stage_output,
                        stage_cpu_output)

import unittest

class TestPartition(unittest.TestCase):
    def test_partition(self):
        from dataset import input_schema, output_schema
        from pprint import pprint

        source_data_arrays, feed_many, feed_once = _partition(
                                    ('utime', 'row'), input_schema())

    def test_construct_staging_areas(self):
        from dataset import input_schema, output_schema

        devices = ['/cpu:0']

        _construct_tensorflow_staging_areas(input_schema(),
            output_schema(), ('utime', 'row'), devices)


    def test_construct_tensorflow_expression(self):
        from dataset import input_schema, output_schema

        devices = ['/cpu:0']
        slvr_cfg = {'polarisation_type': 'linear'}

        feed_data = _construct_tensorflow_staging_areas(input_schema(),
            output_schema(), ('utime', 'row'), devices)

        expr = _construct_tensorflow_expression(feed_data, slvr_cfg,
                                                        devices[0], 0)

if __name__ == "__main__":
    unittest.main()