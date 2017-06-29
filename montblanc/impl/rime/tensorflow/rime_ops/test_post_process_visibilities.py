import itertools
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestPostProcessVisibilities(unittest.TestCase):
    """ Tests the PostProcessVisibilities operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()

        # Load the custom operation library
        #self.rime = tf.load_op_library('rime.so')
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_post_process_visibilities(self):
        """ Test the PostProcessVisibilities operator """

        # List of type constraints for testing this operator
        type_permutations = [
            [np.float32, np.complex64],
            [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_post_process_visibilities(FT, CT)

    def _impl_test_post_process_visibilities(self, FT, CT):
        """ Implementation of the PostProcessVisibilities operator test """

        ntime, nbl, na, nchan = 100, 21, 7, 16

        rf = lambda *a, **kw: np.random.random(*a, **kw).astype(FT)
        rc = lambda *a, **kw: rf(*a, **kw) + 1j*rf(*a, **kw).astype(CT)

        # Create input variables
        antenna1 = np.random.randint(low=0, high=na,
            size=[ntime, nbl]).astype(np.int32)
        antenna2 = np.random.randint(low=0, high=na,
            size=[ntime, nbl]).astype(np.int32)
        direction_independent_effects = rc(size=[ntime, na, nchan, 4])
        flag = np.random.randint(low=0, high=2,
            size=[ntime, nbl, nchan, 4]).astype(np.uint8)
        weight = rf(size=[ntime, nbl, nchan, 4])
        base_vis = rc(size=[ntime, nbl, nchan, 4])
        model_vis = rc(size=[ntime, nbl, nchan, 4])
        observed_vis = rc(size=[ntime, nbl, nchan, 4])

        # Argument list
        np_args = [antenna1, antenna2, direction_independent_effects, flag, weight,
            base_vis, model_vis, observed_vis]
        # Argument string name list
        arg_names = ['antenna1', 'antenna2', 'direction_independent_effects',
            'flag', 'weight', 'base_vis', 'model_vis', 'observed_vis']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.post_process_visibilities(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU visiblities and chi squared
            cpu_vis, cpu_X2 = S.run(cpu_op)

            # Compare against the gpu visibilities and chi squared values
            for gpu_vis, gpu_X2 in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_vis, gpu_vis))
                self.assertTrue(np.allclose(cpu_X2, gpu_X2))

if __name__ == "__main__":
    unittest.main()