import collections

import attr
import numpy as np
from attrdict import AttrDict

try:
    import cytoolz as toolz
except ImportError:
    import toolz
import six
import tensorflow as tf

from montblanc.src_types import source_var_types

from montblanc.rime.staging_area_wrapper import create_staging_area_wrapper
from montblanc.rime.queue_dataset import (TensorQueue, QueueDataset)


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

def __construct_tensorflow_expression(feed_data, slvr_cfg, device, dev_id):
    """ Constructs a tensorflow expression for computing the RIME """
    local_cpu = feed_data.local_cpu
    local_compute = feed_data.local_compute

    polarisation_type = slvr_cfg['polarisation_type']

    # Create ops for copying from the CPU to compute staging areas

    # Feed Once Staging Area
    _, data = local_cpu.feed_once.get(local_cpu.feed_once_key,
                                    name="cpu_feed_once_peek")
    stage_feed_once = local_compute.feed_once[dev_id].put(
                                    local_cpu.feed_once_key, data,
                                    name="compute_feed_once_put")

    with tf.control_dependencies([stage_feed_once]):
#    with tf.control_dependencies([]):
        feed_once_key, feed_once_data =  local_compute.feed_once[dev_id].get(
                                    local_cpu.feed_once_key,
                                    name="compute_feed_once_peek")

    # Feed Many Staging Area
    feed_many_key, data = local_cpu.feed_many.get(local_cpu.feed_many_key,
                                    name="cpu_feed_many_get")
    stage_feed_many = local_compute.feed_many[dev_id].put(feed_many_key, data,
                                    name="compute_feed_many_put")

    # Pull RIME inputs out of the feed many staging_area
    # for the relevant device
    with tf.control_dependencies([stage_feed_many]):
#    with tf.control_dependencies([]):
            feed_many_key, feed_many_data = local_compute.feed_many[dev_id].get(
                                    local_cpu.feed_many_key,
                                    name="compute_feed_many_get")

    # Dictionary of inputs merged from feed once and feed many
    D = AttrDict(toolz.merge(feed_once_data, feed_many_data))

    # Get internal data for this computation
    _, I = local_cpu.feed_internal.get_to_attrdict(local_cpu.feed_many_key,
                                    name="compute_feed_internal_key")

    stage_source_loops = []

    for src_type in source_var_types().keys():
        key_attr = "%s_keys" % src_type
        keys = getattr(I, key_attr)

        # How many chunks should be fed?
        nsrc_chunks = tf.cast(tf.shape(keys)[0], tf.int64)

        def cond(chunk):
            return tf.less(chunk, nsrc_chunks)

        def body(chunk):
            key, data = local_cpu.sources[src_type].get(keys[chunk],
                                        name="cpu_%s_get" % src_type)

            feed_src_chunk = local_compute.sources[dev_id][src_type].put(
                                        key, data,
                                        name="compute_%s_put" % src_type)

            # Create a dependency on the  put operation
            with tf.control_dependencies([feed_src_chunk]):
                return [chunk + 1]

        # Depend on the previous while loop, if it exists
        try:
            deps = [stage_source_loops[-1]]
        except IndexError:
            deps= []

#        with tf.control_dependencies([]):
        with tf.control_dependencies(deps):
            loop = tf.while_loop(cond, body, [tf.constant(0,dtype=tf.int64)],
                                 parallel_iterations=1)

        stage_source_loops.append(loop)

 #   with tf.control_dependencies([]):
    with tf.control_dependencies([D.values()[0]]):
        stage_source_data = tf.group(*stage_source_loops)

    # Infer chunk dimensions
    with tf.device(device):
        # Infer chunk dimensions
        model_vis_shape = tf.shape(D.data)
        nvrow, nchan, npol = [model_vis_shape[i] for i in range(3)]

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

        with tf.device('/cpu:0'):
            C = I.point_keys[chunk]

        _, S = point_sources.get_to_attrdict(C, name="point_get")

        # Get source count for this chunk
        nsrc = tf.shape(S.point_lm)[0]

        ant_jones, sgn_brightness = antenna_jones(S.point_lm,
            S.point_stokes, S.point_alpha, S.point_ref_freq)
        shape = tf.ones(shape=[nsrc,nvrow,nchan], dtype=FT)
        coherencies = rime.sum_coherencies(D.time_index,
            D.antenna1, D.antenna2,
            shape, ant_jones, sgn_brightness, coherencies)

        return coherencies, chunk + 1

    def gaussian_body(coherencies, chunk):
        """ Accumulate coherencies for gaussian source batch """
        gaussian_sources = local_compute.sources[dev_id]['gaussian']

        with tf.device('/cpu:0'):
            C = I.gaussian_keys[chunk]

        _, S = gaussian_sources.get_to_attrdict(C, name="gauss_get")

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

        with tf.device('/cpu:0'):
            C = I.sersic_keys[chunk]

        _, S = sersic_sources.get_to_attrdict(C, name="sersic_get")

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
        zero = tf.constant(0, dtype=tf.int32)
        base_coherencies = tf.zeros_like(D.data, optimize=True)

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
            D.weight, D.base_vis, summed_coherencies, D.data)

        # Stage output in the compute output staging area
        stage_output = local_compute.output.put(feed_many_key,
                            { 'model_vis': model_vis,
                             'chi_squared': chi_squared })

    # Create ops for shifting output from compute staging area
    # to CPU staging area
    with tf.control_dependencies([stage_output]):
        out_key, out_data = local_compute.output.get(feed_many_key)
        stage_cpu_output = local_cpu.output.put(out_key, out_data)

    with tf.control_dependencies([stage_cpu_output]):
        _, output_data = local_cpu.output.get(out_key)

    ComputeNodes = attr.make_class("ComputeNodes", ["stage_feed_many",
                                                    "stage_feed_once",
                                                    "stage_source_data",
                                                    "stage_output",
                                                    "stage_cpu_output",
                                                    "model_vis",
                                                    "chi_squared"])

    # Return Compute operations
    return ComputeNodes(stage_feed_many,
                        stage_feed_once,
                        stage_source_data,
                        stage_output,
                        stage_cpu_output,
                        output_data['model_vis'],
                        output_data['chi_squared'])

QueueDatasetDetails = attr.make_class('QueueDatasetDetails',
                                        ['queue',
                                        'dataset',
                                        'iterator',
                                        'next_op',
                                        'put_op',
                                        'destroy_buffer_op',
                                        'placeholders'])

def _create_queue_dataset_details(feed_data, device):
    """
    Creates a queue dataset for the given ``feed_data``
    and ``device`` and returns an object encapsulating
    the details for inserting data into the queue and
    retrieving data from the dataset's associated iterator.

    Parameters
    ----------
    feed_data : dict

    device : str or :class:`tf.DeviceSpec`
        tensorflow device

    Returns
    -------
    :class:`QueueDatasetDetails`
        Contains queue, dataset, iterator objects, as well
        as operations for inserting into the queue (and dataset)
        and the iterator next op.
    """
    from tensorflow.contrib.data.python.ops import prefetching_ops
    from tensorflow.python.data.ops import iterator_ops
    from tensorflow.python.ops import resource_variable_ops
    from tensorflow.python.framework import function
    from tensorflow.python.data.util import nest

    # Work out the shapes and data types handled by the
    # queue (and dataset)
    dtypes = {k: v['dtype'] for k, v in feed_data.items()}
    shapes = {k: [None]*len(v['dims']) for k, v in feed_data.items()}

    # Create the queue and a put operation with associated
    # placeholders for insertion into the queue
    queue = TensorQueue(dtypes, shapes)
    placeholders = {k: tf.placeholder(dtypes[k], shapes[k])
                                for k in feed_data.keys()}
    put = queue.put(placeholders)

    # Now create the queue dataset, associated iterator and next op
    ds = QueueDataset(queue)
    it = ds.make_initializable_iterator()
    next_ = it.get_next()

    # TODO(sjperkins)
    # Replace the following section of code with
    # https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/data/prefetch_to_device
    # which should arrive in tensorflow 1.8

    # Use a prefetch buffer if the device
    # on which the graph executes is a GPU
    if device.device_type == "GPU":
        @function.Defun(tf.string)
        def _remote_fn(h):
            # TODO(sjperkins)
            # function_buffering_resource does not yet seem
            # to support nested structures. Flatten nested
            # structures in types and shapes,
            # then reconstruct nested structures lower down
            # with nest.pack_sequeunce_as
            flat_types = tuple(nest.flatten(ds.output_types))
            flat_shapes = tuple(nest.flatten(ds.output_shapes))

            remote_iterator = iterator_ops.Iterator.from_string_handle(
                h, flat_types, flat_shapes)

            return remote_iterator.get_next()

        # Prefetch from this device
        target = tf.constant('/CPU:0')

        with tf.device(device):
            buf_resource_handle = prefetching_ops.function_buffering_resource(
                f=_remote_fn,
                target_device=target,
                string_arg=it.string_handle(),
                buffer_size=1,
                thread_pool_size=1,
                shared_name="cpu_gpu")

        with tf.device(device):
            flat_types = tuple(nest.flatten(ds.output_types))
            next_ = prefetching_ops.function_buffering_resource_get_next(
                function_buffer_resource=buf_resource_handle,
                output_types=flat_types)

            # Repack next_ back into a structure output by the dataset
            # (and expected by the user)
            next_ = nest.pack_sequence_as(ds.output_types, next_)

        destroy_buf_op = resource_variable_ops.destroy_resource_op(
                    buf_resource_handle, ignore_lookup_error=True)
    else:
        destroy_buf_op = None

    return QueueDatasetDetails(queue, ds, it, next_, put,
                                destroy_buf_op, placeholders)

def _construct_tensorflow_expression(cfg, device):
    """
    Construct a tensorflow expression for the given
    configuration ``cfg`` and tensorflow device ``device``
    """

    from montblanc.impl.rime.tensorflow.dataset import (input_schema,
                                                        internal_schema)
    # Promote string device specifiers to tf.DeviceSpec
    if isinstance(device, six.string_types):
        device = tf.DeviceSpec.from_string(device)

    # Partition input arrays
    (source_data_arrays,
        feed_many,
        feed_once) = _partition(('utime', 'vrow'), input_schema())

    feed_multiple = toolz.merge(feed_once, feed_many, internal_schema())

    # Create the graph
    with tf.Graph().as_default() as graph:
        multiple_dataset = _create_queue_dataset_details(feed_multiple, device)

        source_staging_areas = {k: create_staging_area_wrapper('%s_cpu' % k,
                                    v.keys(), input_schema(),
                                    ordered=True, device=device)
                            for k, v in source_data_arrays.items()}

        inputs = multiple_dataset.next_op

        def point_body(points, lm):
            key = inputs['point_keys'][points]
            staging_area = source_staging_areas['point']
            _, point_inputs = staging_area.get(key, name="point_get")
            print point_inputs['point_lm']
            lm = lm + tf.reduce_sum(point_inputs['point_lm'], axis=0)
            lm.set_shape((2,))

            return points+1, lm

        def gaussian_body(gaussians, lm):
            key = inputs['gaussian_keys'][gaussians]
            staging_area = source_staging_areas['gaussian']
            _, gaussian_inputs = staging_area.get(key)
            lm = lm + tf.reduce_sum(gaussian_inputs['gaussian_lm'], axis=0)
            lm.set_shape((2,))

            return gaussians+1, lm

        def sersic_body(sersics, lm):
            key = inputs['sersic_keys'][sersics]
            staging_area = source_staging_areas['sersic']
            _, sersic_inputs = staging_area.get(key)
            lm = lm + tf.reduce_sum(sersic_inputs['sersic_lm'], axis=0)

            lm.set_shape((2,))

            return sersics+1, lm

        with tf.device(device):
            zero_lm = tf.constant([0.0,0.0], dtype=tf.float64)
            zero_index = tf.constant(0, dtype=tf.int32)

            npsrc = tf.shape(inputs['point_keys'])[0]
            _, plm = tf.while_loop(lambda p, lm: tf.less(p, npsrc),
                            point_body, [zero_index, zero_lm])

            ngsrc = tf.shape(inputs['gaussian_keys'])[0]
            _, glm = tf.while_loop(lambda g, lm: tf.less(g, ngsrc),
                            gaussian_body, [zero_index, zero_lm])

            nssrc = tf.shape(inputs['sersic_keys'])[0]
            _, slm = tf.while_loop(lambda s, lm: tf.less(s, nssrc),
                            sersic_body, [zero_index, zero_lm])

            result = (plm, glm, slm)

        pprint(inputs)

    TensorflowExpression = attr.make_class("TensorflowExpression",
        ["multiple_dataset", "source_staging_areas", "graph",
        "result"])

    return TensorflowExpression(multiple_dataset, source_staging_areas,
                                graph, result)

import unittest
from dataset import input_schema
from pprint import pprint

class TestPartition(unittest.TestCase):
    def test_partition(self):
        (source_data_arrays, feed_many,
            feed_once) = _partition(('utime', 'vrow'), input_schema())

    def test_construct_tensorflow_expression(self):
        cfg = {'polarisation_type': 'linear'}

        def _dummy_data(ph):
            """ Generate some dummy data given a tensorflow placeholder """
            shape = tuple(2 if s is None else s for s in ph.shape.as_list())
            return np.zeros(shape, dtype=ph.dtype.as_numpy_dtype())

        # Test with available devices (CPU + GPU)
        with tf.Session() as S:
            devices = [d.name for d in S.list_devices()]

        # Test each device separately
        for device in devices:
            expr = _construct_tensorflow_expression(cfg, device)

            mds = expr.multiple_dataset
            mphs = mds.placeholders

            with tf.Session(graph=expr.graph) as S:
                # Initialise the iterator
                S.run(expr.multiple_dataset.iterator.initializer)

                def _feed_source(source, keys):
                    src = expr.source_staging_areas[source]
                    lm_str = '%s_lm' % source
                    lm_ph = src.placeholders[src.fed_arrays.index(lm_str)]

                    feed_dict = {ph: _dummy_data(ph) for ph in src.placeholders }

                    for i, key in enumerate(keys):
                        feed_dict.update({src.put_key_ph: key})
                        feed_dict.update({lm_ph: np.full((10,2), i+1)})
                        S.run(src.put_op, feed_dict=feed_dict)

                # Feed some dummy data into the queue
                feed_dict = {ph: _dummy_data(ph) for ph in mphs.values()}
                feed_dict.update({mphs['point_keys'] : [0, 1, 2]})
                _feed_source('point', [0, 1, 2])
                feed_dict.update({mphs['gaussian_keys'] : [0, 1, 2]})
                _feed_source('gaussian', [0, 1, 2])
                feed_dict.update({mphs['sersic_keys'] : [0, 1, 2]})
                _feed_source('sersic', [0, 1, 2])

                S.run(expr.multiple_dataset.put_op, feed_dict=feed_dict)


                print S.run(expr.result)

if __name__ == "__main__":
    unittest.main()
