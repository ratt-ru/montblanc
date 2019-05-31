import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def radec_to_lm(radec, phase_centre):
        pc_ra, pc_dec = phase_centre
        sin_d0 = np.sin(pc_dec)
        cos_d0 = np.cos(pc_dec)

        da = radec[:, 0] - pc_ra
        sin_da = np.sin(da)
        cos_da = np.cos(da)

        sin_d = np.sin(radec[:, 1])
        cos_d = np.cos(radec[:, 1])

        lm = np.empty_like(radec)
        lm[:, 0] = cos_d*sin_da
        lm[:, 1] = sin_d*cos_d0 - cos_d*sin_d0*cos_da

        return lm


class TestRadecToLm(unittest.TestCase):
    """ Tests the RadecToLm operator """

    def setUp(self):
        # Load the custom operation library
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_radec_to_lm(self):
        """ Test the RadecToLm operator """
        # List of type constraint for testing this operator
        type_permutations = [np.float32, np.float64]

        # Run test with the type combinations above
        for FT in type_permutations:
            self._impl_test_radec_to_lm(FT)

    def _impl_test_radec_to_lm(self, FT):
        """ Implementation of the RadecToLm operator test """

        # Create input variables
        nsrc = 10
        radec = np.random.random(size=[nsrc, 2]).astype(FT)
        phase_centre = np.random.random(size=[2]).astype(FT)

        # Argument list
        np_args = [radec, phase_centre]
        # Argument string name list
        arg_names = ['radec', 'phase_centre']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.radec_to_lm(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            cpu_result = S.run(cpu_op)
            assert np.allclose(cpu_result, radec_to_lm(*np_args))

            for gpu_result in S.run(gpu_ops):
                assert np.allclose(cpu_result, gpu_result)

if __name__ == "__main__":
    unittest.main()
