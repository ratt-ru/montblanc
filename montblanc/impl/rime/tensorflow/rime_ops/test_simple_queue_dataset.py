import threading
import unittest

import numpy as np
import tensorflow as tf

from montblanc.impl.rime.tensorflow.queue_dataset import (TensorQueue,
                                                        QueueDataset)

class TestQueueTensorDataset(unittest.TestCase):

    def test_queue_tensor_dataset(self):
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
