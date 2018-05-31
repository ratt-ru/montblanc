import threading
import unittest

import numpy as np
import tensorflow as tf

from montblanc.impl.rime.tensorflow.map_dataset import (TensorMap,
                                                        MapDataset)

class TestMapTensorDataset(unittest.TestCase):

    def test_dataset_in_graph_while_loop(self):
        N = 12
        nkeys = 6

        with tf.Session() as S:
            devices = [dev.name for dev in S.list_devices()]

        for device in devices:
            with tf.Graph().as_default() as graph:
                key_ph = tf.placeholder(tf.int64, name="key", shape=())
                value_ph = tf.placeholder(tf.int64, name="value", shape=())
                keys_ph = tf.placeholder(tf.int64, name="keys", shape=(None,1))

                dtypes = value_ph.dtype

                tensor_map = TensorMap(dtypes, tf.TensorShape([]))
                key_ds = tf.data.Dataset.from_tensor_slices(keys_ph)
                ds = MapDataset(key_ds, tensor_map)
                ds = ds.apply(tf.contrib.data.prefetch_to_device(device, buffer_size=1))

                insert_op = tensor_map.insert(key_ph, value_ph)
                close_op = tensor_map.close()

                it = ds.make_initializable_iterator()

                def cond(i, s):
                    return tf.less(i, tf.size(keys_ph))

                def body(i, s):
                    v = it.get_next()
                    s = s + v
                    return i+1, s

                deps = [it.initializer]

                with tf.control_dependencies(deps):
                    loop = tf.while_loop(cond, body,
                        [tf.convert_to_tensor(0, dtype=tf.int32),
                        tf.convert_to_tensor(0, dtype=tf.int64)])

                global_init_op = tf.global_variables_initializer()

            with tf.Session(graph=graph) as S:
                S.run(global_init_op)

                for i in range(N):
                    keys = i*nkeys + np.arange(nkeys, dtype=np.int64)

                    for key in keys:
                        S.run(insert_op, feed_dict={key_ph: key, value_ph: i})

                    keys =  keys.reshape((nkeys,1))
                    S.run([it.initializer, loop], feed_dict={keys_ph: keys})

                S.run(close_op)

    def test_numpy_conversion(self):
        with tf.Graph().as_default() as graph:
            ck = tf.placeholder(dtype=tf.int64)
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}
            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            tensor_map = TensorMap(dtypes)
            ds = MapDataset(tf.data.Dataset.range(2,3), tensor_map)

            insert_op = tensor_map.insert(2, {'i': np.int64(23),
                                'sub' : {'f': hundred_floats}})
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])
            S.run(insert_op)

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)


    def test_nest_dtype_only(self):
        with tf.Graph().as_default() as graph:
            ck = tf.placeholder(dtype=tf.int64)
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}

            tensor_map = TensorMap(dtypes)
            ds = MapDataset(tf.data.Dataset.range(2,3), tensor_map)

            insert_op = tensor_map.insert(ck, {'i': ci, 'sub' : {'f': cf}})
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(insert_op, feed_dict={ck: 2, ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)

    def test_nest_dtypes_and_shapes(self):
        with tf.Graph().as_default() as graph:
            ck = tf.placeholder(dtype=tf.int64)
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            # dtypes and shapes must have the same structure
            dtypes = { 'i': ci.dtype, 'sub' : {'f': cf.dtype}}
            shapes = { 'i': None, 'sub' : {'f': [10, 10]}}

            tensor_map = TensorMap(dtypes)
            ds = MapDataset(tf.data.Dataset.range(2,3), tensor_map)

            insert_op = tensor_map.insert(ck, {'i': ci, 'sub' : {'f': cf}})
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10,10), 2.0, dtype=np.float64)

            S.run(insert_op, feed_dict={ck: 2, ci: 23, cf: hundred_floats})

            result = S.run(next_op)
            self.assertTrue(np.all(hundred_floats == result['sub']['f']))
            self.assertTrue(23 == result['i'])
            S.run(close_op)

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
