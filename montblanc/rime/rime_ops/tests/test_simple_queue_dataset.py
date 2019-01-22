import threading
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.data.experimental import prefetch_to_device, copy_to_device

from montblanc.rime.queue_dataset import (TensorQueue, QueueDataset)


class TestQueueTensorDataset(unittest.TestCase):

    def test_multiple_thread_enqueue_and_dequeue(self):
        with tf.Session() as S:
            devices = [dev.name for dev in S.list_devices()
                       if 'XLA' not in dev.name]

        with tf.Graph().as_default() as graph:
            pi = tf.placeholder(dtype=tf.int64)
            dtypes = {'i': pi.dtype}

            queue = TensorQueue(dtypes)
            ds = QueueDataset(queue)

            with tf.device('/CPU:0'):
                put_op = queue.put({'i': pi})

            close_op = queue.close()

            datasets = [ds.apply(copy_to_device(target_device=device))
                        for device in devices]
            dataset = [ds.prefetch(1) for ds in datasets]
            iterators = [ds.make_initializable_iterator() for ds in datasets]
            next_ops = [it.get_next() for it in iterators]

            global_init_op = tf.global_variables_initializer()

        print_lock = threading.Lock()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op] + [it.initializer for it in iterators])

            def _enqueue(n):
                for i in range(1, n+1):
                    S.run(put_op, feed_dict={pi: i})

                S.run(close_op)

            def _dequeue(op):
                while True:
                    try:
                        print(S.run(op))
                    except tf.errors.OutOfRangeError:
                        return

            enqueue_thread = threading.Thread(target=_enqueue, args=(10,))
            dequeue_threads = [threading.Thread(target=_dequeue, args=(op,))
                               for op in next_ops]

            enqueue_thread.start()

            for t in dequeue_threads:
                t.start()

            enqueue_thread.join()

            for t in dequeue_threads:
                t.join()



    def test_numpy_conversion(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}
            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

            queue = TensorQueue(dtypes)
            ds = QueueDataset(queue)

            put_op = queue.put({'i': np.int64(23),
                                'sub': {'f': hundred_floats}})
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])
            S.run(put_op)

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)

    def test_nest_dtype_only(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}

            queue = TensorQueue(dtypes)
            ds = QueueDataset(queue)

            put_op = queue.put({'i': ci, 'sub': {'f': cf}})
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

            S.run(put_op, feed_dict={ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)

    def test_nest_dtypes_and_shapes(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            # dtypes and shapes must have the same structure
            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}
            shapes = {'i': None, 'sub': {'f': [10, 10]}}

            queue = TensorQueue(dtypes, shapes)
            ds = QueueDataset(queue)

            put_op = queue.put({'i': ci, 'sub': {'f': cf}})
            close_op = queue.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

            S.run(put_op, feed_dict={ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)

    def test_dataset_in_graph_while_loop(self):
        with tf.Session() as S:
            devices = [dev.name for dev in S.list_devices()
                       if 'XLA' not in dev.name]

        for device in devices:
            with tf.Graph().as_default() as graph:
                ci = tf.placeholder(dtype=tf.int64)
                cf = tf.placeholder(dtype=tf.float64)

                dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}
                queue = TensorQueue(dtypes)
                ds = QueueDataset(queue)

                put_op = queue.put({'i': ci, 'sub': {'f': cf}})
                close_op = queue.close()

                ds = ds.apply(prefetch_to_device(device, buffer_size=1))
                it = ds.make_initializable_iterator()
                next_op = it.get_next()

                global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])
            N = 12

            def _enqueue(n):
                for i in range(1, n+1):
                    S.run(put_op, feed_dict={ci: [i]*i, cf: [i]*i})

                S.run(close_op)

            t = threading.Thread(target=_enqueue, args=(N,))
            t.start()

            for i in range(1, N+1):
                data = [i]*i

                np_ints = np.asarray(data, dtype=np.int64)
                np_floats = np.asarray(data, dtype=np.float64)

                result = S.run(next_op)
                tf_ints, tf_floats = result['i'], result['sub']['f']

                self.assertTrue(np.all(np_ints == tf_ints))
                self.assertTrue(np.all(np_floats == tf_floats))

            with self.assertRaises(tf.errors.OutOfRangeError):
                S.run(next_op)

            t.join()

    def test_basic(self):
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
                for i in range(1, n+1):
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

            with self.assertRaises(tf.errors.OutOfRangeError):
                S.run(next_op)

            t.join()


if __name__ == "__main__":
    unittest.main()
