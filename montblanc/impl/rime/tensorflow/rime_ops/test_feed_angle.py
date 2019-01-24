import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


class TestFeedAngle(unittest.TestCase):
    """ Tests the FeedAngle operator """

    def setUp(self):
        # Load the custom operation library
        self.rime = tf.load_op_library('rime.so')
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_feed_angle(self):
        """ Test the FeedAngle operator """
        # List of type constraint for testing this operator
        type_permutations = [
            [np.float32, np.complex64],
            [np.float32, np.complex128],
            [np.float64, np.complex64],
            [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_feed_angle(FT, CT)

    def _impl_test_feed_angle(self, FT, CT):
        """ Implementation of the FeedAngle operator test """

        # Create input variables
        feed_angle = np.random.random(size=[1]).astype(FT)
        parallactic_angle = np.random.random(size=[1, 1]).astype(FT)

        # Argument list
        np_args = [feed_angle, parallactic_angle]
        # Argument string name list
        arg_names = ['feed_angle', 'parallactic_angle']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.feed_angle(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            S.run(cpu_op)
            S.run(gpu_ops)


if __name__ == "__main__":
    unittest.main()
