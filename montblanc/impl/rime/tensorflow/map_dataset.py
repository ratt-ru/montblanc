import tensorflow as tf

from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

from montblanc.impl.rime.tensorflow.tensorflow_ops import (simple_map_dataset as mds,
                                                        dataset_map_handle,
                                                        dataset_map_insert,
                                                        dataset_map_close,
                                                        dataset_map_size)

class TensorMap(object):
    """
    A Map of tensors.
    """

    def __init__(self, dtypes, shapes=None, shared_name=None):
        """
        Constructs a simple map accepting ``put`` operations
        of tensors with the specified ``dtypes`` and ``shapes``.

        ``dtypes`` and ``shapes`` may be either tuples, or
        nested dict/tuple structures. For example:

        ..code-block:: python

            ci = tf.placeholder(tf.int64)
            cf = tf.placeholder(tf.float64)

            dtypes = { 'a': ci.dtype, 'sub' : { 'b': cf.dtype } }
            shapes = { 'a': (), 'sub' : { 'b': (10,10) } }

            map = TensorMap(dtypes, shapes)
            put_op = map.put( {'a': ci, 'sub' : { 'b': cf } })

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
            Shared resource name if this Map is to be
            shared amongst multiple tensorflow Sesssions.
        """
        with ops.name_scope("tensor_map") as scope:
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
        self.handle = dataset_map_handle(flat_dtypes, flat_shapes,
                                           name=scope, shared_name=shared_name)

    def insert(self, key, tensors, name=None):
        if name is None:
            name = "tensor_map_insert"

        nest.assert_same_structure(tensors, self.output_types)
        flat_dtypes = nest.flatten(self.output_types)
        key = ops.convert_to_tensor(key, dtype=tf.int64, name="%s_key" % name)
        tensors = tuple(ops.convert_to_tensor(t, dtype=dt,
                                    name="%s_component_%i" % (name, i))
            for i, (t, dt)
            in enumerate(zip(nest.flatten(tensors), flat_dtypes)))

        return dataset_map_insert(self.handle, key, tensors, name=name)

    def close(self, name=None):
        return dataset_map_close(self.handle, name=name)

    def size(self, name=None):
        return dataset_map_size(self.handle, name=name)

class MapDataset(tf.data.Dataset):
  """
  A `Dataset` consuming elements from a `TensorMap`
  """
  def __init__(self, key_dataset, tensor_map, name=None):
    super(MapDataset, self).__init__()
    self._key_dataset = key_dataset
    self._map = tensor_map
    self._name = name

  def _as_variant_tensor(self):
    return mds(self._key_dataset._as_variant_tensor(),
                self._map.handle, name=self._name)

  @property
  def output_shapes(self):
    return self._map.output_shapes

  @property
  def output_types(self):
    return self._map.output_types

  @property
  def output_classes(self):
    return self._map.output_classes
