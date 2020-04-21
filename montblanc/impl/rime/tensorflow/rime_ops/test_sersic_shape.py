import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def rf(*s):
    return np.random.random(size=s)


class TestSersicShape(unittest.TestCase):
    """ Tests the SersicShape operator """
    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_sersic_shape(self):
        for FT in [np.float32, np.float64]:
            self._impl_test_sersic_shape(FT)

    def _impl_test_sersic_shape(self, FT):
        ngsrc, ntime, na, nchan = 10, 15, 7, 16
        nbl = na*(na-1)//2

        np_uvw = rf(ntime, na, 3).astype(FT)
        np_ant1, np_ant2 = [np.int32(x) for x in np.triu_indices(na, 1)]
        np_ant1, np_ant2 = (np.tile(np_ant1, ntime).reshape(ntime, nbl),
                            np.tile(np_ant2, ntime).reshape(ntime, nbl))
        np_frequency = np.linspace(1.4e9, 1.5e9, nchan).astype(FT)
        np_sersic_params = rf(3, ngsrc)*np.array([1.0, 1.0, np.pi/648000],
                                                 dtype=FT)[:, np.newaxis]
        np_sersic_params = np_sersic_params.astype(dtype=FT)

        assert np_ant1.shape == (ntime, nbl), np_ant1.shape
        assert np_ant2.shape == (ntime, nbl), np_ant2.shape
        assert np_frequency.shape == (nchan,)

        np_args = [np_uvw, np_ant1, np_ant2, np_frequency, np_sersic_params]
        arg_names = ["uvw", "ant1", "ant2", "frequency", "sersic_params"]

        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.sersic_shape(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            cpu_sersic_shape = S.run(cpu_op)

            for gpu_sersic_shape in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_sersic_shape,
                                            gpu_sersic_shape))


if __name__ == "__main__":
    unittest.main()
