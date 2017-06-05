import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestSumCoherencies(unittest.TestCase):
    """ Tests the SumCoherencies operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()

        # Load the custom operation library
        #self.rime = tf.load_op_library('rime.so')
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_sum_coherencies(self):
        """ Test the SumCoherencies operator """

        # List of type constraints for testing this operator
        type_permutations = [
            [np.float32, np.complex64],
            [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_sum_coherencies(FT, CT)

    def _impl_test_sum_coherencies(self, FT, CT):
        """ Implementation of the SumCoherencies operator test """

        rf = lambda *a, **kw: np.random.random(*a, **kw).astype(FT)
        rc = lambda *a, **kw: rf(*a, **kw) + 1j*rf(*a, **kw).astype(CT)

        nsrc, ntime, na, nchan = 10, 15, 7, 16
        nbl = na*(na-1)//2

        np_ant1, np_ant2 = map(lambda x: np.int32(x), np.triu_indices(na, 1))
        np_ant1, np_ant2 = (np.tile(np_ant1, ntime).reshape(ntime, nbl),
            np.tile(np_ant2, ntime).reshape(ntime,nbl))
        np_shape = rf(size=(nsrc, ntime, nbl, nchan))
        np_ant_jones = rc(size=(nsrc, ntime, na, nchan, 4))
        np_sgn_brightness = np.random.randint(0, 3, size=(nsrc, ntime), dtype=np.int8) - 1
        np_base_coherencies =  rc(size=(ntime, nbl, nchan, 4))

        # Argument list
        np_args = [np_ant1, np_ant2, np_shape, np_ant_jones,
            np_sgn_brightness, np_base_coherencies]
        # Argument string name list
        arg_names = ['antenna1', 'antenna2', 'shape', 'ant_jones',
            'sgn_brightness', 'base_coherencies']
        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.sum_coherencies(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU coherencies
            cpu_coherencies = S.run(cpu_op)

            # Compare against the GPU coherencies
            for gpu_coherencies in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_coherencies, gpu_coherencies))

if __name__ == "__main__":
    unittest.main()

