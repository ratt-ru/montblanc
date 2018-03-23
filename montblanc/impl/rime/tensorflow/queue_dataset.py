import tensorflow as tf

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
# from tensorflow.python.eager import context
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
# from tensorflow.python.ops import array_ops
# from tensorflow.python.ops import gen_dataset_ops
# from tensorflow.python.ops import gen_io_ops
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import script_ops
# from tensorflow.python.util import deprecation
# from tensorflow.python.util.tf_export import tf_export

from montblanc.impl.rime.tensorflow.tensorflow_ops import (queue_dataset as qds,
                                                        dataset_queue_handle,
                                                        dataset_queue_enqueue,
                                                        dataset_queue_close)

class TensorQueue(object):
    """
    A Queue of tensors.
    """
    def __init__(self, dtypes, shapes=None, shared_name=None):
        with ops.name_scope("tensor_queue") as scope:
            if isinstance(dtypes, tuple):
                pass
            elif isinstance(dtypes, list):
                dtypes = tuple(dtypes)
            else:
                dtypes = (dtypes,)

            self.output_types = dtypes

            if shapes is not None:
                assert len(shapes) == len(dtypes)

                self.output_shapes = shapes
            else:
                self.output_shapes = tuple(tensor_shape.unknown_shape()
                                        for dt in dtypes)

        self.output_classes = tuple(ops.Tensor for dt in dtypes)
        self.handle = dataset_queue_handle(dtypes, self.output_shapes,
                                           name=scope, shared_name=shared_name)

    def put(self, tensors, name=None):
        return dataset_queue_enqueue(self.handle, tensors, name=name)

    def close(self, name=None):
        return dataset_queue_close(self.handle, name=name)

class QueueDataset(tf.data.Dataset):
  """
  A `Dataset` consuming elements from a `TensorQueue`
  """
  def __init__(self, queue, name=None):
    super(QueueDataset, self).__init__()
    self._queue = queue
    self._name = name

  def _as_variant_tensor(self):
    return qds(self._queue.handle, name=self._name)

  @property
  def output_shapes(self):
    return self._queue.output_shapes

  @property
  def output_types(self):
    return self._queue.output_types

  @property
  def output_classes(self):
    return self._queue.output_classes
