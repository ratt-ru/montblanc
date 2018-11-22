from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data import prefetch_to_device


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

    blah = inputs['data']

    return blah
