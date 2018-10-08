from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.data import prefetch_to_device

import montblanc.impl.rime.tensorflow.tensorflow_ops as ops
from montblanc.impl.rime.tensorflow.map_dataset import MapDataset
from montblanc.impl.rime.tensorflow.utils import source_decorator


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
    gaussian_input_map = source_input_maps["gaussian_inputs"]
    # Create a key dataset from the set of __point_keys__
    point_key_ds = tf.data.Dataset.from_tensor_slices(
                        inputs["__point_keys__"])
    gaussian_key_ds = tf.data.Dataset.from_tensor_slices(
                        inputs["__gaussian_keys__"])
    # Create a point inputs dataset, retrieving point data from
    # the point input map per key
    point_inputs_ds = MapDataset(point_key_ds, point_input_map)
    gaussian_inputs_ds = MapDataset(gaussian_key_ds, gaussian_input_map)

    # Apply GPU prefetch to point data
    if device.device_type == "GPU":
        xform = prefetch_to_device(device, buffer_size=1)
        point_inputs_ds = point_inputs_ds.apply(xform)
        gaussian_inputs_ds = gaussian_inputs_ds.apply(xform)

    # Create an iterator over point source data
    point_inputs_it = point_inputs_ds.make_initializable_iterator()
    gaussian_inputs_it = gaussian_inputs_ds.make_initializable_iterator()

    model_vis_shape = tf.shape(inputs['data'])
    nrow, nchan, ncorr = map(model_vis_shape.__getitem__, range(3))
    FT, CT = inputs['frequency'].dtype, inputs['data'].dtype

    @source_decorator("point")
    def point_body(points, base_coherencies):
        point_inputs = point_inputs_it.get_next()

        complex_phase = ops.phase(point_inputs['point_lm'],
                                  inputs['uvw'],
                                  inputs['frequency'],
                                  lm_schema="(source,(l,m))",
                                  uvw_schema="(row,(u,v,w))",
                                  CT=CT)

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

    @source_decorator("gaussian")
    def gaussian_body(gaussians, base_coherencies):
        gaussian_inputs = gaussian_inputs_it.get_next()

        complex_phase = ops.phase(gaussian_inputs['gaussian_lm'],
                                  inputs['uvw'],
                                  inputs['frequency'],
                                  lm_schema="(source,(l,m))",
                                  uvw_schema="(row,(u,v,w))",
                                  CT=CT)

        brightness = ops.brightness(gaussian_inputs['gaussian_stokes'],
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

        return gaussians+1, coherencies

    # point dataset iterator  must be initialised
    deps = [point_inputs_it.initializer]

    with tf.device(device), tf.control_dependencies(deps):
        base_coherencies = tf.zeros_like(inputs['data'], optimize=True)
        npsrc = tf.shape(inputs['__point_keys__'])[0]
        _, summed_coherencies = tf.while_loop(lambda p, coh: tf.less(p, npsrc),
                                              point_body,
                                              [0, base_coherencies])

        ngsrc = tf.shape(inputs['__gaussian_keys__'])[0]
        _, sum_coherencies = tf.while_loop(lambda g, coh: tf.less(g, ngsrc),
                                           gaussian_body,
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
