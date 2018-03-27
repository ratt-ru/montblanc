import threading
import unittest

import numpy as np
import tensorflow as tf

from montblanc.impl.rime.tensorflow.map_dataset import (TensorMap,
                                                        MapDataset)

class TestMapTensorDataset(unittest.TestCase):

    def __test_numpy_conversion(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}
            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            map = TensorMap(dtypes)
            ds = MapDataset(map)

            insert_op = map.put({'i': np.int64(23),
                                'sub' : {'f': hundred_floats}})
            close_op = map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])
            S.run(insert_op)

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])


    def __test_nest_dtype_only(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}

            map = TensorMap(dtypes)
            ds = MapDataset(map)

            insert_op = map.put({'i': ci, 'sub' : {'f': cf}})
            close_op = map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(insert_op, feed_dict={ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])

    def __test_nest_dtypes_and_shapes(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            # dtypes and shapes must have the same structure
            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}
            shapes = { 'i': None, 'sub' : {'f': [10, 10]}}

            map = TensorMap(dtypes, shapes)
            ds = MapDataset(map)

            insert_op = map.put({'i': ci, 'sub' : {'f': cf}})
            close_op = map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(insert_op, feed_dict={ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])

    def test_basic(self):
        N = 12

        with tf.Graph().as_default() as graph:
            ck = tf.placeholder(dtype=tf.int64)
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            tensor_map = TensorMap((tf.int64, tf.float64))
            key_ds = tf.data.Dataset.range(1, N+1)
            ds = MapDataset(key_ds, tensor_map)
            ds = ds.map(lambda i, f: (i+1, f*2), num_parallel_calls=3)
            ds = ds.prefetch(1)

            insert_op = tensor_map.insert(ck, (ci, cf))
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            def _insert(n):
                for i in  range(1, n+1):
                    S.run(insert_op, feed_dict={ck: i, ci: [i]*i, cf: [i]*i})

                S.run(close_op)

            t = threading.Thread(target=_insert, args=(N,))
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
