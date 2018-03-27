from os.path import join as pjoin

import pkg_resources

import tensorflow as tf

# Load standard/development version of rime tensorflow library?
if False:
    # Installed library location
    _rime_lib_path = pkg_resources.resource_filename("montblanc", "ext")
else:
    # Development library location
    _rime_lib_path = pkg_resources.resource_filename("montblanc",
                            pjoin('impl', 'rime', 'tensorflow', 'rime_ops'))

_rime_so = tf.load_op_library(pjoin(_rime_lib_path, 'rime.so'))

# RIME operators for export
_export_ops = ["b_sqrt", "create_antenna_jones", "e_beam", "feed_rotation",
                "gauss_shape", "parallactic_angle_sin_cos", "phase",
                "post_process_visibilities", "sersic_shape",
                "sum_coherencies"]
# Queue Dataset operators for export
_export_ops += ["dataset_queue_handle", "dataset_queue_enqueue",
                "dataset_queue_close", "simple_queue_dataset"]

# Map Dataset operators for export
_export_ops += ["dataset_map_handle", "dataset_map_insert",
                "dataset_map_close", "simple_map_dataset"]

print dir(_rime_so)

# Store ops in this module
globals().update({n: getattr(_rime_so, n) for n in _export_ops})

