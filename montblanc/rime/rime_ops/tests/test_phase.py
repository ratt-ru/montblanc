import unittest
from itertools import product

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from montblanc.rime.tensorflow_ops import phase as phase_op

lightspeed = 299792458.


def complex_phase_numpy(lm, uvw, frequency):
    """ Compute complex phase using numpy """

    # Set up slicing depending on whether a row based uvw
    # scheme is used
    flm = lm.reshape(-1, 2)
    fuvw = uvw.reshape(-1, 3)

    l = flm[:, None, 0:1]
    m = flm[:, None, 1:2]

    u = fuvw[None, :, 0:1]
    v = fuvw[None, :, 1:2]
    w = fuvw[None, :, 2:3]

    freq = frequency[None, None, :]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    real_phase = -2*np.pi*1j*(l*u + m*v + n*w)*freq/lightspeed
    shape = lm.shape[:-1] + uvw.shape[:-1] + frequency.shape
    return np.exp(real_phase).reshape(shape)


class TestComplexPhase(unittest.TestCase):
    """ Tests the ComplexPhase operator """

    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_complex_phase(self):
        """ Test the ComplexPhase operator """

        # List of type constraints for testing this operator
        types = [[np.float32, np.complex64],
                 [np.float64, np.complex128]]

        lm_shapes = [(10,), (3, 4,), (5, 8, 10)]
        uvw_shapes = [(30,), (10, 2), (4, 3, 2)]

        for (FT, CT), lms, uvws in product(types, lm_shapes, uvw_shapes):
            self._impl_test_complex_phase(FT, CT, lms, uvws)

    def _impl_test_complex_phase(self, FT, CT, lm_shape, uvw_shape):
        """ Implementation of the ComplexPhase operator test """

        nchan = 16

        # Set up our numpy input arrays
        lm = np.random.random(size=lm_shape + (2,)).astype(FT)*0.1
        uvw = np.random.random(size=uvw_shape + (3,)).astype(FT)
        frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=FT)

        np_args = [lm, uvw, frequency]
        arg_names = ["lm", "uvw", "frequency"]

        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, op, *args, **kwargs):
            """ Pin operation to device """
            with tf.device(device):
                return op(*args, **kwargs)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', phase_op, *tf_args, CT=CT)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, phase_op, *tf_args, CT=CT)
                   for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU complex phase
            cpu_cplx_phase = S.run(cpu_op)

            # Compare vs numpy
            np_cplx_phase = complex_phase_numpy(lm, uvw, frequency)
            self.assertTrue(np.allclose(np_cplx_phase, cpu_cplx_phase))

            # Compare vs GPU
            for gpu_cplx_phase in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_cplx_phase, gpu_cplx_phase))

if __name__ == "__main__":
    unittest.main()
