import unittest

import tensorflow as tf

class TestQueuedTensorDataset(unittest.TestCase):

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib("./rime.so")

    def test_queued_tensor_dataset(self):

        # TODO(sjperkins).
        # Move QueuedTensorDataset into own python file.
        from tensorflow.python.data.ops import iterator_ops
        from tensorflow.python.data.util import nest
        from tensorflow.python.data.util import random_seed
        from tensorflow.python.data.util import sparse
        from tensorflow.python.eager import context
        from tensorflow.python.framework import constant_op
        from tensorflow.python.framework import dtypes
        from tensorflow.python.framework import function
        from tensorflow.python.framework import ops
        from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
        from tensorflow.python.framework import tensor_shape
        from tensorflow.python.framework import tensor_util
        from tensorflow.python.ops import array_ops
        from tensorflow.python.ops import gen_dataset_ops
        from tensorflow.python.ops import gen_io_ops
        from tensorflow.python.ops import math_ops
        from tensorflow.python.ops import script_ops
        from tensorflow.python.util import deprecation
        from tensorflow.python.util.tf_export import tf_export

        # HACK
        thang = self

        print thang.rime.queued_tensor_dataset.__doc__

        class QueuedTensorDataset(tf.data.Dataset):
          """A `Dataset` with a single element, viz. a nested structure of tensors."""

          def __init__(self, dtypes, shapes=None):
            """See `Dataset.from_tensors()` for details."""
            super(QueuedTensorDataset, self).__init__()

            with ops.name_scope("tensors"):
                if isinstance(dtypes, tuple):
                    pass
                elif isinstance(dtypes, list):
                    dtypes = tuple(dtypes)
                else:
                    dtypes = (dtypes,)

                self._output_types = dtypes

                if shapes is not None:
                    assert len(shapes) == len(dtypes)

                    self._output_shapes = shapes
                else:
                    self._output_shapes = tuple(tensor_shape.scalar() for dt in self._output_types)

            self._output_classes = tuple(ops.Tensor for dt in self._output_types)

          def _as_variant_tensor(self):
            return thang.rime.queued_tensor_dataset(
                Toutput_types=self._output_types,
                Toutput_shapes=self._output_shapes)

          @property
          def output_shapes(self):
            return self._output_shapes

          @property
          def output_types(self):
            return self._output_types

          @property
          def output_classes(self):
            return self._output_classes

        #ds = tf.data.Dataset.range(100)
        ds = QueuedTensorDataset((tf.int64,tf.float64))
        print ds
        it = ds.make_initializable_iterator()
        next_op = it.get_next()

        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            print "pre-init"
            S.run([init_op, it.initializer])
            print "post-init"

            print S.run(next_op)
            print S.run(next_op)
            print S.run(next_op)

if __name__ == "__main__":
    unittest.main()
