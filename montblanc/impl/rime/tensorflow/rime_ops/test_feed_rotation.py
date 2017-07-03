import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestFeedRotation(unittest.TestCase):
    """ Tests the FeedRotation operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_feed_rotation(self):
        """ Test the FeedRotation operator """
        # List of type constraint for testing this operator
        type_permutations = [
            [np.float32, np.complex64, 'linear'],
            [np.float64, np.complex128, 'linear'],
            [np.float32, np.complex64, 'circular'],
            [np.float64, np.complex128, 'circular'],
        ]

        # Run test with the type combinations above
        for FT, CT, feed_type in type_permutations:
            self._impl_test_feed_rotation(FT, CT, feed_type)

    def _impl_test_feed_rotation(self, FT, CT, feed_type):
        """ Implementation of the FeedRotation operator test """

        # Create input variables
        ntime, na = 10, 7

        parallactic_angle = np.random.random(size=[ntime,na]).astype(FT)
        parallactic_angle_sin = np.sin(parallactic_angle)
        parallactic_angle_cos = np.cos(parallactic_angle)

        # Argument list
        np_args = [parallactic_angle_sin, parallactic_angle_cos]
        # Argument string name list
        arg_names = ['parallactic_angle_sin', 'parallactic_angle_cos']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.feed_rotation(*tf_args,
                    CT=CT, feed_type=feed_type)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            cpu_feed_rotation = S.run(cpu_op)

            for gpu_feed_rotation in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_feed_rotation, gpu_feed_rotation))

if __name__ == "__main__":
    unittest.main()