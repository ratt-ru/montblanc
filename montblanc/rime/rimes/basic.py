from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import montblanc.rime.tensorflow_ops as ops
import tensorflow as tf
from montblanc.rime.utils import source_context
from tensorflow.contrib.data import prefetch_to_device

from montblanc.rime.map_dataset import MapDataset

should_prefetch = False
buffer_size = 1


def create_tf_expr(cfg, device, input_ds, source_input_maps):
    polarisation_type = cfg['polarisation_type']
    debug = cfg.get('debug', False)

    # Apply GPU prefetch to input dataset
    if should_prefetch and device.device_type == "GPU":
        xform = prefetch_to_device(device, buffer_size=buffer_size)
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
    if should_prefetch and device.device_type == "GPU":
        xform = prefetch_to_device(device, buffer_size=buffer_size)
        point_inputs_ds = point_inputs_ds.apply(xform)

    # Create an iterator over point source data
    point_inputs_it = point_inputs_ds.make_initializable_iterator()

    model_vis_shape = tf.shape(inputs['data'])
    nrow, nchan, ncorr = map(model_vis_shape.__getitem__, range(3))
    FT, CT = inputs['frequency'].dtype, inputs['data'].dtype

    @source_context("point")
    def point_body(points, base_coherencies):
        point_inputs = point_inputs_it.get_next()

        complex_phase = ops.phase(point_inputs['point_lm'],
                                  inputs['uvw'],
                                  inputs['frequency'],
                                  lm_schema="(source,(l,m))",
                                  uvw_schema="(row,(u,v,w))",
                                  CT=CT)

        phase_msg = ("Check that '1 - l**2  - m**2 >= 0' holds "
                     "for all your lm coordinates. This is required "
                     "for 'n = sqrt(1 - l**2 - m**2) - 1' "
                     "to be finite.")

        phase_real = tf.check_numerics(tf.real(complex_phase), phase_msg)
        phase_imag = tf.check_numerics(tf.imag(complex_phase), phase_msg)

        brightness = ops.brightness(point_inputs['point_stokes'],
                                    stokes_schema="(source,corr)",
                                    CT=CT)

        bl_jones = ops.jones_multiply([complex_phase, brightness],
                                      schemas=["(source,row,chan)",
                                               "(source,corr)"],
                                      output_schema="(source,row,chan,corr)",
                                      FT=FT)

        coherencies = ops.sum_coherencies(
                        inputs['time_index'],
                        inputs['antenna1'],
                        inputs['antenna2'],
                        [],
                        [bl_jones],
                        [],
                        [base_coherencies],
                        FT=FT, CT=CT)

        return points+1, coherencies

    # point dataset iterator  must be initialised
    deps = [point_inputs_it.initializer]

    with tf.device(device), tf.control_dependencies(deps):
        base_coherencies = tf.zeros_like(inputs['data'], optimize=True)
        npsrc = tf.shape(inputs['__point_keys__'])[0]
        _, summed_coherencies = tf.while_loop(lambda p, coh: tf.less(p, npsrc),
                                              point_body,
                                              [0, base_coherencies])

        # Post process visibilities to produce
        # model visibilities and chi squared
        model_vis, chi_squared = ops.post_process_visibilities(
            inputs["time_index"], inputs["antenna1"], inputs["antenna2"],
            inputs["direction_independent_effects"], inputs["flag"],
            inputs["weight"], base_coherencies,
            summed_coherencies, inputs["data"])

        result = (model_vis, chi_squared)

    return result
