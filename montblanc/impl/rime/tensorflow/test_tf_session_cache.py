import unittest

import tensorflow as tf

from montblanc.impl.rime.tensorflow.tf_session_cache import tf_session_cache
from montblanc.impl.rime.tensorflow.tf_graph import (
                        _construct_tensorflow_staging_areas,
                        _construct_tensorflow_expression)
from montblanc.impl.rime.tensorflow.dataset import (
                        input_schema, output_schema)


def _create_tensorflow_graph():
    """ Create a tensorflow graph """
    devices = ['/cpu:0']
    slvr_cfg = {'polarisation_type': 'linear'}

    with tf.Graph().as_default() as graph:
        feed_data = _construct_tensorflow_staging_areas(input_schema(),
            output_schema(), ('utime', 'vrow'), devices)

        expr = _construct_tensorflow_expression(feed_data, slvr_cfg,
                                                        devices[0], 0)

        init_op = tf.global_variables_initializer()

    return graph, init_op, expr, feed_data

class TestTensorflowSessionCache(unittest.TestCase):
    def test_tf_session_cache(self):
        graph, init_op, expr, feed_data = _create_tensorflow_graph()

        with tf_session_cache().open(tf.Session, "", graph=graph) as S:
            S.run(init_op)

        self.assertTrue(tf_session_cache().size() == 1)

        with tf_session_cache().open(tf.Session, "", graph=graph) as S:
            S.run(init_op)

        self.assertTrue(tf_session_cache().size() == 1)

        graph, init_op, expr, feed_data = _create_tensorflow_graph()

        with tf_session_cache().open(tf.Session, "", graph=graph) as S:
            S.run(init_op)

        self.assertTrue(tf_session_cache().size() == 2)

if __name__ == "__main__":
    unittest.main()


