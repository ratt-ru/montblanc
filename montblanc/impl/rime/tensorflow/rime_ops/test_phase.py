import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

def complex_phase_numpy(lm, uvw, frequency):
    """ Compute complex phase using numpy """

    lightspeed = 299792458.
    nsrc, _ = lm.shape
    narow, _ = uvw.shape
    nchan, = frequency.shape

    l = lm[:,None,None,0]
    m = lm[:,None,None,1]
    u = uvw[None,:,None,0]
    v = uvw[None,:,None,1]
    w = uvw[None,:,None,2]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    real_phase = -2*np.pi*1j*(l*u + m*v + n*w)*frequency/lightspeed
    return np.exp(real_phase)

class TestComplexPhase(unittest.TestCase):
    """ Tests the ComplexPhase operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib("rime.so")
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']


    def test_complex_phase(self):
        """ Test the ComplexPhase operator """

        # List of type constraints for testing this operator
        type_permutations = [[np.float32, np.complex64],
                            [np.float64, np.complex128]]

        for FT, CT in type_permutations:
            self._impl_test_complex_phase(FT, CT)

    def _impl_test_complex_phase(self, FT, CT):
        """ Implementation of the ComplexPhase operator test """
        nsrc, narow, nchan = 10, 15*16, 16

        # Set up our numpy input arrays
        lm = np.random.random(size=(nsrc,2)).astype(FT)*0.1
        uvw = np.random.random(size=(narow,3)).astype(FT)
        frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=FT)

        np_args = [lm, uvw, frequency]
        arg_names = ["lm", "uvw", "frequency"]

        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.phase(*tf_args, CT=CT)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU ejones
            cpu_cplx_phase = S.run(cpu_op)

            np_cplx_phase = complex_phase_numpy(lm, uvw, frequency)

            self.assertTrue(np.allclose(np_cplx_phase, cpu_cplx_phase))

            for gpu_cplx_phase in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_cplx_phase, gpu_cplx_phase))

if __name__ == "__main__":
    unittest.main()
