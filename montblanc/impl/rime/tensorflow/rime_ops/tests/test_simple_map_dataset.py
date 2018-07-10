import threading
import unittest

import numpy as np
import tensorflow as tf

from tensorflow.contrib.data import prefetch_to_device

from montblanc.impl.rime.tensorflow.map_dataset import (TensorMap,
                                                        MapDataset)


class TestMapTensorDataset(unittest.TestCase):

    def test_dataset_in_graph_while_loop(self):
        N = 12
        nkeys = 7

        with tf.Session() as S:
            devices = [dev.name for dev in S.list_devices()]

        for device in devices:
            with tf.Graph().as_default() as graph:
                key_ph = tf.placeholder(tf.int64, name="key",
                                        shape=())
                value_ph = tf.placeholder(tf.int64, name="value",
                                          shape=())
                keys_ph = tf.placeholder(tf.int64, name="keys",
                                         shape=(None, 1))

                dtypes = value_ph.dtype

                tensor_map = TensorMap(dtypes, tf.TensorShape([]), store=True)
                key_ds = tf.data.Dataset.from_tensor_slices(keys_ph)
                ds = MapDataset(key_ds, tensor_map)
                ds = ds.apply(prefetch_to_device(device, buffer_size=1))

                insert_op = tensor_map.insert(key_ph, value_ph)
                clear_key_ph = tf.placeholder(tf.int64, name="clear_keys",
                                              shape=(None,))
                clear_op = tensor_map.clear(keys=clear_key_ph)
                close_op = tensor_map.close()
                keys_op = tensor_map.keys()
                size_op = tensor_map.size()

                it = ds.make_initializable_iterator()

                def cond(i, s):
                    return tf.less(i, tf.size(keys_ph))

                def body(i, s):
                    v = it.get_next()
                    n = tf.add(s, v)
                    return i+1, n

                deps = [it.initializer]

                with tf.control_dependencies(deps):
                    with tf.device(device):
                        loop_vars = [tf.constant(0, dtype=tf.int32),
                                     tf.constant(0, dtype=tf.int64)]
                        loop = tf.while_loop(cond, body, loop_vars,
                                             parallel_iterations=1)

                global_init_op = tf.global_variables_initializer()

            with tf.Session(graph=graph) as S:
                S.run(global_init_op)

                for i in range(N):
                    keys = i*nkeys + np.arange(nkeys, dtype=np.int64)
                    clear_keys = keys

                    for j, key in enumerate(keys):
                        S.run(insert_op, feed_dict={
                                            key_ph: key,
                                            value_ph: j+i})

                    map_keys = np.sort(S.run(keys_op))
                    self.assertTrue(np.all(map_keys == keys))

                    keys = keys.reshape((nkeys, 1))
                    _, vals = S.run([it.initializer, loop],
                                    feed_dict={keys_ph: keys})

                    # Clear the keys out in two batches
                    clear_keys_1 = clear_keys[:len(clear_keys)//2]
                    clear_keys_2 = clear_keys[len(clear_keys)//2:]
                    S.run(clear_op, feed_dict={clear_key_ph: clear_keys_1})
                    remaining_keys = np.sort(S.run(keys_op))
                    self.assertTrue((remaining_keys == clear_keys_2).all())
                    self.assertTrue(S.run(size_op) == len(clear_keys_2))
                    S.run(clear_op, feed_dict={clear_key_ph: clear_keys_2})
                    self.assertTrue(S.run(size_op) == 0)

                S.run(close_op)

    def test_numpy_conversion(self):
        with tf.Graph().as_default() as graph:
            ci = tf.placeholder(dtype=tf.int64)
            cf = tf.placeholder(dtype=tf.float64)

            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}
            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

            tensor_map = TensorMap(dtypes)
            ds = MapDataset(tf.data.Dataset.range(2, 3), tensor_map)

            insert_op = tensor_map.insert(2, {'i': np.int64(23),
                                              'sub': {'f': hundred_floats}})
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

            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}

            tensor_map = TensorMap(dtypes)
            ds = MapDataset(tf.data.Dataset.range(2, 3), tensor_map)

            insert_op = tensor_map.insert(ck, {'i': ci, 'sub': {'f': cf}})
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

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
            dtypes = {'i': ci.dtype, 'sub': {'f': cf.dtype}}
            shapes = {'i': None, 'sub': {'f': [10, 10]}}

            tensor_map = TensorMap(dtypes, shapes)
            ds = MapDataset(tf.data.Dataset.range(2, 3), tensor_map)

            insert_op = tensor_map.insert(ck, {'i': ci, 'sub': {'f': cf}})
            close_op = tensor_map.close()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            hundred_floats = np.full((10, 10), 2.0, dtype=np.float64)

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
            size_op = tensor_map.size()

            it = ds.make_initializable_iterator()
            next_op = it.get_next()

            global_init_op = tf.global_variables_initializer()

        with tf.Session(graph=graph) as S:
            S.run([global_init_op, it.initializer])

            def _insert(n):
                for i in range(1, n+1):
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

            with self.assertRaises(tf.errors.OutOfRangeError):
                S.run(next_op)

            self.assertTrue(S.run(size_op) == 0)

            t.join()


if __name__ == "__main__":
    unittest.main()
