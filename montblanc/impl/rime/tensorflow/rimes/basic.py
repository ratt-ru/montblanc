from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from pprint import pprint

import tensorflow as tf

from tensorflow.contrib.data import prefetch_to_device

import montblanc.impl.rime.tensorflow.tensorflow_ops as ops
from montblanc.impl.rime.tensorflow.map_dataset import MapDataset


def create_tf_expr(cfg, device, input_ds, source_input_maps):
    polarisation_type = cfg['polarisation_type']
    debug = cfg.get('debug', False)

    # Apply GPU prefetch to input dataset
    if device.device_type == "GPU":
        xform = prefetch_to_device(device, buffer_size=1)
        input_ds = input_ds.apply(xform)

    # Create iterator
    inputs_it = input_ds.make_initializable_iterator()
    # Get inputs from the iterator
    inputs = inputs_it.get_next()

    # Obtain the tensor map for point inputs
    point_input_map = source_input_maps["point_inputs"]
    # Create a key dataset from the set of __point_keys__
    point_key_ds = tf.data.Dataset.from_tensor_slices(inputs["__point_keys__"])
    # Create a point inputs dataset, retrieving point data from
    # the point input map per key
    point_inputs_ds = MapDataset(point_key_ds, point_input_map)

    # Apply GPU prefetch to point data
    if device.device_type == "GPU":
        xform = prefetch_to_device(device, buffer_size=1)
        point_inputs_ds = point_inputs_ds.apply(xform)

    # Create an iterator over point source data
    point_inputs_it = point_inputs_ds.make_initializable_iterator()

    model_vis_shape = tf.shape(inputs['data'])
    nrow, nchan, ncorr = map(model_vis_shape.__getitem__, range(3))
    FT, CT = inputs['frequency'].dtype, inputs['data'].dtype

    # Feed rotation is used within the while loop bodies
    # Create the expression for it upfront
    with tf.device(device):
        pa_sin, pa_cos = ops.parallactic_angle_sin_cos(
                            inputs['parallactic_angles'])
        feed_rotation = ops.feed_rotation(pa_sin, pa_cos, CT=CT,
                                          feed_type=polarisation_type)

    def antenna_jones(lm, stokes, alpha, ref_freq):
        """
        Compute the jones terms for each antenna.

        `lm`, `stokes`, `alpha` and `ref_freq` are the source variables.
        """
        # Compute the square root of the brightness matrix
        # (as well as the sign)
        bsqrt, sgn_brightness = ops.b_sqrt(stokes, alpha,
            inputs['frequency'], ref_freq, CT=CT,
            polarisation_type=polarisation_type)

        # Check for nans/infs in the bsqrt
        bsqrt_msg = ("Check that your stokes parameters "
                    "satisfy I**2 >= Q**2 + U**2 + V**2. "
                    "Montblanc performs a cholesky decomposition "
                    "of the brightness matrix and the above must "
                    "hold for this to produce valid values.")

        bsqrt_real = tf.check_numerics(tf.real(bsqrt), bsqrt_msg)
        bsqrt_imag = tf.check_numerics(tf.imag(bsqrt), bsqrt_msg)

        # Create dependencies on checks if debugging
        deps = [] if not debug else [phase_real, phase_imag,
                                     bsqrt_real, bsqrt_imag]

        # Combine the brightness square root, complex phase,
        # feed rotation and beam dde's
        with tf.control_dependencies(deps):
            antenna_jones = ops.create_antenna_jones([bsqrt],
                                            [],
                                            [feed_rotation],
                                            [],
                                            FT=FT, CT=CT)

        return antenna_jones, sgn_brightness


    def point_body(points, base_coherencies):
        point_inputs = point_inputs_it.get_next()
        lm = point_inputs['point_lm']
        nsrc = tf.shape(lm)[0]

        # Point source shape terms are unity
        shape = tf.ones(shape=[nsrc,nrow,nchan], dtype=FT)

        ant_jones, sgn_brightness = antenna_jones(lm,
                                        point_inputs['point_stokes'],
                                        point_inputs['point_alpha'],
                                        point_inputs['point_ref_freq'])

        complex_phase = ops.phase(lm,
                            inputs['uvw'],
                            inputs['frequency'],
                            uvw_schema="(row,(u,v,w))",
                            CT=CT)

        phase_msg = ("Check that '1 - l**2  - m**2 >= 0' holds "
                    "for all your lm coordinates. This is required "
                    "for 'n = sqrt(1 - l**2 - m**2) - 1' "
                    "to be finite.")

        phase_real = tf.check_numerics(tf.real(complex_phase), phase_msg)
        phase_imag = tf.check_numerics(tf.imag(complex_phase), phase_msg)


        coherencies = ops.sum_coherencies(
                        inputs['time_index'],
                        inputs['antenna1'],
                        inputs['antenna2'],
                        shape,
                        ant_jones,
                        [sgn_brightness],
                        [complex_phase],
                        [base_coherencies])

        return points+1, coherencies


    # point dataset iterator  must be initialised
    deps = [point_inputs_it.initializer]

    with tf.device(device), tf.control_dependencies(deps):
        base_coherencies = tf.zeros_like(inputs['data'], optimize=True)
        npsrc = tf.shape(inputs['__point_keys__'])[0]
        _, summed_coherencies = tf.while_loop(lambda p, coh: tf.less(p, npsrc),
                                              point_body,
                                              [0, base_coherencies])



        # Post process visibilities to produce model visibilities and chi squared
        model_vis, chi_squared = ops.post_process_visibilities(
            inputs["time_index"], inputs["antenna1"], inputs["antenna2"],
            inputs["direction_independent_effects"], inputs["flag"],
            inputs["weight"], base_coherencies,
            summed_coherencies, inputs["data"])

        result = (model_vis, chi_squared)

    return result