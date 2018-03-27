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

from montblanc.impl.rime.tensorflow.tensorflow_ops import (simple_queue_dataset as qds,
                                                        dataset_queue_handle,
                                                        dataset_queue_enqueue,
                                                        dataset_queue_close)

class TensorQueue(object):
    """
    A Queue of tensors.
    """

    def __init__(self, dtypes, shapes=None, shared_name=None):
        """
        Constructs a simple queue accepting ``put`` operations
        of tensors with the specified ``dtypes`` and ``shapes``.

        ``dtypes`` and ``shapes`` may be either tuples, or
        nested dict/tuple structures. For example:

        ..code-block:: python

            ci = tf.placeholder(tf.int64)
            cf = tf.placeholder(tf.float64)

            dtypes = { 'a': ci.dtype, 'sub' : { 'b': cf.dtype } }
            shapes = { 'a': (), 'sub' : { 'b': (10,10) } }

            queue = TensorQueue(dtypes, shapes)
            put_op = queue.put( {'a': ci, 'sub' : { 'b': cf } })

            with tf.Session() as S:
                S.run(put_op, feed_dict={ci: 2, cf: np.ones((10,10))})

        Parameters
        ----------
        dtypes : nested dicts or nested tuples
            A nested collection of dicts or tuples
            containing dtypes
        shapes : nested dicts or nested tuples
            A nested collection of dicts or tuples
            containing shapes associated with ``dtypes``.
            Must have the same structure as ``dtypes``
        shared_name : str, optional
            Shared resource name if this Queue is to be
            shared amongst multiple tensorflow Sesssions.
        """
        with ops.name_scope("tensor_queue") as scope:
            flat_dtypes = nest.flatten(dtypes)

            if shapes is None:
                uk = tensor_shape.unknown_shape()
                flat_shapes = tuple(uk for dt in flat_dtypes)
            else:
                shapes = nest.map_structure(tensor_shape.as_shape, shapes)
                flat_shapes = nest.flatten(shapes)

            flat_classes = tuple(ops.Tensor for dt in flat_dtypes)

        self.output_types = dtypes
        self.output_shapes = nest.pack_sequence_as(dtypes, flat_shapes)
        self.output_classes = nest.pack_sequence_as(dtypes, flat_classes)
        self.handle = dataset_queue_handle(flat_dtypes, flat_shapes,
                                           name=scope, shared_name=shared_name)

    def put(self, tensors, name=None):
        nest.assert_same_structure(tensors, self.output_types)
        flat_dtypes = nest.flatten(self.output_types)
        tensors = tuple(
            ops.convert_to_tensor(t, dtype=dt, name="component_%i"%i)
            for i, (t, dt)
            in enumerate(zip(nest.flatten(tensors), flat_dtypes)))

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
