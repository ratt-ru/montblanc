from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import montblanc.rime.tensorflow_ops as ops
import tensorflow as tf
from montblanc.rime.utils import source_context
from tensorflow.data.experimental import prefetch_to_device

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
    gaussian_input_map = source_input_maps["gaussian_inputs"]
    sersic_input_map = source_input_maps["sersic_inputs"]
    # Create a key dataset from the set of __point_keys__
    point_key_ds = tf.data.Dataset.from_tensor_slices(
                        inputs["__point_keys__"])
    gaussian_key_ds = tf.data.Dataset.from_tensor_slices(
                        inputs["__gaussian_keys__"])
    sersic_key_ds = tf.data.Dataset.from_tensor_slices(
                        inputs["__sersic_keys__"])
    # Create a point inputs dataset, retrieving point data from
    # the point input map per key
    point_inputs_ds = MapDataset(point_key_ds, point_input_map)
    gaussian_inputs_ds = MapDataset(gaussian_key_ds, gaussian_input_map)
    sersic_inputs_ds = MapDataset(sersic_key_ds, sersic_input_map)

    # Apply GPU prefetch to source data
    if should_prefetch and device.device_type == "GPU":
        point_xform = prefetch_to_device(device, buffer_size=buffer_size)
        gaussian_xform = prefetch_to_device(device, buffer_size=buffer_size)
        sersic_xform = prefetch_to_device(device, buffer_size=buffer_size)

        point_inputs_ds = point_inputs_ds.apply(point_xform)
        gaussian_inputs_ds = gaussian_inputs_ds.apply(gaussian_xform)
        sersic_inputs_ds = sersic_inputs_ds.apply(sersic_xform)

    # Create an iterator over point source data
    point_inputs_it = point_inputs_ds.make_initializable_iterator()
    gaussian_inputs_it = gaussian_inputs_ds.make_initializable_iterator()
    sersic_inputs_it = sersic_inputs_ds.make_initializable_iterator()

    model_vis_shape = tf.shape(inputs['data'])
    nrow, nchan, ncorr = map(model_vis_shape.__getitem__, range(3))
    FT, CT = inputs['frequency'].dtype, inputs['data'].dtype

    @source_context("point")
    def point_body(points, coherencies):
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
                        [coherencies],
                        FT=FT, CT=CT)

        return points+1, coherencies

    @source_context("gaussian")
    def gaussian_body(gaussians, coherencies):
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

        gauss_shape = ops.gauss_shape(inputs['uvw'],
                                      inputs['frequency'],
                                      gaussian_inputs['gauss_params'])

        gauss_shape = tf.cast(gauss_shape, dtype=CT)

        bl_jones = ops.jones_multiply([gauss_shape, complex_phase, brightness],
                                      schemas=["(source,row,chan)",
                                               "(source,row,chan)",
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
                        [coherencies],
                        FT=FT, CT=CT)

        return gaussians+1, coherencies

    @source_context("sersic")
    def sersic_body(sersics, coherencies):
        sersic_inputs = sersic_inputs_it.get_next()

        complex_phase = ops.phase(sersic_inputs['sersic_lm'],
                                  inputs['uvw'],
                                  inputs['frequency'],
                                  lm_schema="(source,(l,m))",
                                  uvw_schema="(row,(u,v,w))",
                                  CT=CT)

        brightness = ops.brightness(sersic_inputs['sersic_stokes'],
                                    stokes_schema="(source,corr)",
                                    CT=CT)

        gauss_shape = ops.sersic_shape(inputs['uvw'],
                                       inputs['frequency'],
                                       sersic_inputs['sersic_params'])

        gauss_shape = tf.cast(gauss_shape, dtype=CT)

        bl_jones = ops.jones_multiply([gauss_shape, complex_phase, brightness],
                                      schemas=["(source,row,chan)",
                                               "(source,row,chan)",
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
                        [coherencies],
                        FT=FT, CT=CT)

        return sersics+1, coherencies

    # Dataset iterators must be initialised
    deps = [inputs_it.initializer,
            point_inputs_it.initializer,
            gaussian_inputs_it.initializer,
            sersic_inputs_it.initializer]
    npsrc = tf.size(inputs['__point_keys__'])
    ngsrc = tf.size(inputs['__gaussian_keys__'])
    nssrc = tf.size(inputs['__sersic_keys__'])

    deps.append(tf.print("Point Chunk Keys:", inputs['__point_keys__']))
    deps.append(tf.print("Gaussian Chunk Keys:", inputs['__gaussian_keys__']))
    deps.append(tf.print("Sersic Chunk Keys:", inputs['__sersic_keys__']))

    with tf.device(device), tf.control_dependencies(deps):
        base_coherencies = tf.zeros_like(inputs['data'], optimize=False)
        _, summed_coherencies = tf.while_loop(lambda p, coh: tf.less(p, npsrc),
                                              point_body,
                                              [0, base_coherencies])

        _, summed_coherencies = tf.while_loop(lambda g, coh: tf.less(g, ngsrc),
                                              gaussian_body,
                                              [0, summed_coherencies])

        _, summed_coherencies = tf.while_loop(lambda s, coh: tf.less(s, nssrc),
                                              sersic_body,
                                              [0, summed_coherencies])

        # Post process visibilities to produce
        # model visibilities and chi squared
        model_vis, chi_squared = ops.post_process_visibilities(
            inputs["time_index"], inputs["antenna1"], inputs["antenna2"],
            inputs["direction_independent_effects"], inputs["flag"],
            inputs["weight"], base_coherencies,
            summed_coherencies, inputs["data"])

        result = (model_vis, chi_squared)

    return result
