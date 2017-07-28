import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestCircularStokesSwap(unittest.TestCase):
    """ Tests the CircularStokesSwap operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_circular_stokes_swap(self):
        """ Test the CircularStokesSwap operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32, np.float64]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_circular_stokes_swap(FT)

    def _impl_test_circular_stokes_swap(self, FT):
        """ Implementation of the CircularStokesSwap operator test """

        nsrc = 65
        ntime = 33
        npol = 4

        # Create input variables
        stokes = np.random.random(size=[nsrc, ntime, npol]).astype(FT)

        # Argument list
        np_args = [stokes]
        # Argument string name list
        arg_names = ['stokes']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.circular_stokes_swap(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            cpu_stokes = S.run(cpu_op)
            for gpu_stokes in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_stokes, gpu_stokes))

if __name__ == "__main__":
    unittest.main()