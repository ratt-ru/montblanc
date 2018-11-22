import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from montblanc.rime.tensorflow_ops import brightness as brightness_op


def numpy_brightness(stokes):
    I = stokes[..., 0]
    Q = stokes[..., 1]
    U = stokes[..., 2]
    V = stokes[..., 3]

    if stokes.dtype == np.float32:
        dtype = np.complex64
    elif stokes.dtype == np.float64:
        dtype = np.complex128
    else:
        raise ValueError("Invalid dtype %s" % stokes.dtype)

    corrs = np.empty_like(stokes, dtype=dtype)

    corrs[..., 0] = I + Q
    corrs[..., 1] = U + V*1j
    corrs[..., 2] = U - V*1j
    corrs[..., 3] = I - Q

    return corrs


class TestBrightness(unittest.TestCase):
    """ Tests the Brightness operator """

    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_brightness(self):
        """ Test the Brightness operator """
        # List of type constraint for testing this operator
        type_permutations = [
            [np.float32, np.complex64],
            [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_brightness(FT, CT)

    def _impl_test_brightness(self, FT, CT):
        """ Implementation of the Brightness operator test """

        # Create input variables
        stokes = np.random.random(size=(100, 64, 4)).astype(FT)

        # Argument list
        np_args = [stokes]
        # Argument string name list
        arg_names = ['stokes']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return brightness_op(*tf_args, CT=CT)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)
            cpu_brightness = S.run(cpu_op)
            np_brightness = numpy_brightness(stokes)

            assert np.allclose(cpu_brightness, np_brightness)

            for gpu_brightness in S.run(gpu_ops):
                assert np.allclose(cpu_brightness, gpu_brightness)

if __name__ == "__main__":
    unittest.main()
