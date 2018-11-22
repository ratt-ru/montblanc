import unittest

import cppimport
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

dsmod = cppimport.imp("montblanc.ext.dataset_mod")

from montblanc.rime.tensorflow_ops import sersic_shape as sersic_shape_op

class TestSersicShape(unittest.TestCase):
    """ Test the Sersic Shape Operator """

    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_sersic_shape(self):
        """ Test the Sersic Shape Operator """

        # List of type constraints for testing this operator
        type_permutations = [
            [np.float32, np.complex64],
            [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_sersic_shape(FT, CT)

    def _impl_test_sersic_shape(self, FT, CT):
        """ Implementation of the Sersic Shape Operator test """

        rf = lambda *a, **kw: np.random.random(*a, **kw).astype(FT)
        rc = lambda *a, **kw: rf(*a, **kw) + 1j*rf(*a, **kw).astype(CT)

        nssrc, ntime, na, nchan = 10, 15, 7, 16
        nbl = na*(na-1)//2

        from montblanc.rime.rime_ops.op_test_utils import random_baselines

        chunks = np.random.random_integers(int(3.*nbl/4.), nbl, ntime)
        nvrow = np.sum(chunks)

        np_uvw, _, _, _ = random_baselines(chunks, na)
        np_uvw = np_uvw.astype(FT)
        np_frequency = np.linspace(1.4e9, 1.5e9, nchan).astype(FT)
        sp_modifier = np.array([[1.0],[1.0],[np.pi/648000]],dtype=FT)
        np_sersic_params = rf((3, nssrc))*sp_modifier

        np_args = [np_uvw, np_frequency, np_sersic_params]
        arg_names = ["uvw", "frequency", "sersic_params"]

        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return sersic_shape_op(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU coherencies
            cpu_shape = S.run(cpu_op)

            # Compare against the GPU coherencies
            for gpu_shape in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_shape, gpu_shape))


if __name__ == "__main__":
    unittest.main()

