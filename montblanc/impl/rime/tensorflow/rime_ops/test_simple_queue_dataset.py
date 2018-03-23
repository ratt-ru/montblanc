import threading
import unittest

import numpy as np
import tensorflow as tf

class TestQueueTensorDataset(unittest.TestCase):

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()

    def test_queue_tensor_dataset(self):

        # TODO(sjperkins).
        # Move QueueDataset into own python file.
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

        class TensorQueue(object):
            def __init__(self, dtypes, shapes=None):
                with ops.name_scope("tensors"):
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
                        self.output_shapes = tuple(tensor_shape.unknown_shape() for dt in self.output_types)

                self.output_classes = tuple(ops.Tensor for dt in self.output_types)
                self.handle = thang.rime.dataset_queue_handle(self.output_types, self.output_shapes)

            def put(self, tensors):
                return thang.rime.dataset_queue_enqueue(self.handle, tensors)

            def close(self):
                return thang.rime.dataset_queue_close(self.handle)

        class QueueDataset(tf.data.Dataset):
          """A `Dataset` consuming elements from a queue"""

          def __init__(self, queue):
            super(QueueDataset, self).__init__()
            self._queue = queue

          def _as_variant_tensor(self):
            return thang.rime.queue_dataset(self._queue.handle)

          @property
          def output_shapes(self):
            return self._queue.output_shapes

          @property
          def output_types(self):
            return self._queue.output_types

          @property
          def output_classes(self):
            return self._queue.output_classes

        N = 12

        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            queue = TensorQueue([tf.int64, tf.float64])
            ds = QueueDataset(queue)
            ds = ds.map(lambda i, f: (i+1, f*2), num_parallel_calls=3)
            ds = ds.prefetch(1)

            put_op = queue.put([ci, cf])
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            def _enqueue(n):
                for i in  range(1, n+1):
                    S.run(put_op, feed_dict={ci: [i]*i, cf: [i]*i})

                S.run(close_op)

            t = threading.Thread(target=_enqueue, args=(N,))
            t.start()

            for i in range(1, N+1):
                data = [i]*i

                np_ints = np.asarray(data, dtype=np.int64)
                np_floats = np.asarray(data, dtype=np.float64)

                tf_ints, tf_floats = S.run(next_op)

                self.assertTrue(np.all(np_ints+1 == tf_ints))
                self.assertTrue(np.all(np_floats*2 == tf_floats))


            with self.assertRaises(tf.errors.OutOfRangeError) as cm:
                S.run(next_op)

            t.join()

if __name__ == "__main__":
    unittest.main()
