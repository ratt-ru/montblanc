import threading
import unittest

import numpy as np
import tensorflow as tf

from montblanc.impl.rime.tensorflow.queue_dataset import (TensorQueue,
                                                        QueueDataset)

class TestQueueTensorDataset(unittest.TestCase):

    def test_queue_tensor_dataset_nest(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}

            queue = TensorQueue(dtypes)
            ds = QueueDataset(queue)

            put_op = queue.put({'i': ci, 'sub' : {'f': cf}})
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            twenty_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(put_op, feed_dict={ci: 23, cf: twenty_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(twenty_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])

        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            # dtypes and shapes must have the same structure
            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}
            shapes = { 'i': None, 'sub' : {'f': [10, 10]}}

            queue = TensorQueue(dtypes, shapes)
            ds = QueueDataset(queue)

            put_op = queue.put({'i': ci, 'sub' : {'f': cf}})
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            twenty_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(put_op, feed_dict={ci: 23, cf: twenty_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(twenty_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])


    def test_queue_tensor_dataset(self):
        N = 12

        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            queue = TensorQueue((tf.int64, tf.float64))
            ds = QueueDataset(queue)
            ds = ds.map(lambda i, f: (i+1, f*2), num_parallel_calls=3)
            ds = ds.prefetch(1)

            put_op = queue.put((ci, cf))
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
