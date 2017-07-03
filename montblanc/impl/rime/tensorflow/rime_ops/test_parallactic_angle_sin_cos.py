import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestParallacticAngleSinCos(unittest.TestCase):
    """ Tests the ParallacticAngleSinCos operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_parallactic_angle_sin_cos(self):
        """ Test the ParallacticAngleSinCos operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32, np.float64]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_parallactic_angle_sin_cos(FT)

    def _impl_test_parallactic_angle_sin_cos(self, FT):
        """ Implementation of the ParallacticAngleSinCos operator test """

        # Create input variables
        ntime = 10
        na = 7

        parallactic_angle = np.random.random(size=[ntime, na]).astype(FT)


        # Argument list
        np_args = [parallactic_angle]
        # Argument string name list
        arg_names = ['parallactic_angle']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.parallactic_angle_sin_cos(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU sincos
            cpu_pa_sin, cpu_pa_cos = S.run(cpu_op)

            # Compare with GPU sincos
            for gpu_pa_sin, gpu_pa_cos in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_pa_sin, gpu_pa_sin))
                self.assertTrue(np.allclose(cpu_pa_cos, gpu_pa_cos))

if __name__ == "__main__":
    unittest.main()