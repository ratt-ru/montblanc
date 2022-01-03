import os
import unittest
import timeit

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.python.client import device_lib

lightspeed = 299792458.


def complex_phase(lm, uvw, frequency):
    """
    Compute the complex phase from lm, uvw and frequency expressions
    """

    # Get the dynamic shape of input tensors
    lm_shape = tf.shape(input=lm)
    uvw_shape = tf.shape(input=uvw)
    frequency_shape = tf.shape(input=frequency)

    # The shapes are themselves tensors
    nsrc = lm_shape[0]
    ntime, na = uvw_shape[0], uvw_shape[1]
    nchan = frequency_shape[0]

    # Define some constants
    one = tf.constant(1.0, dtype=dtype)
    minus_two_pi_over_C = tf.constant(-2.0*np.pi/lightspeed, dtype=dtype)

    # Reshape now so that we get broadcasting in later operations
    # Need to pack list since list contains tensors, e.g. nsrc
    l = tf.reshape(lm[:, 0], tf.stack([nsrc, 1, 1, 1]))
    m = tf.reshape(lm[:, 1], tf.stack([nsrc, 1, 1, 1]))

    u = tf.reshape(uvw[:, :, 0], tf.stack([1, ntime, na, 1]))
    v = tf.reshape(uvw[:, :, 1], tf.stack([1, ntime, na, 1]))
    w = tf.reshape(uvw[:, :, 2], tf.stack([1, ntime, na, 1]))

    frequency = tf.reshape(frequency, tf.stack([1, 1, 1, nchan]))

    n = tf.sqrt(one - l**2 - m**2) - one

    # Outer product l*u + m*v * n*w
    phase = tf.convert_to_tensor(value=l*u + m*v + n*w, name='real_phase')

    # Multiply in constants
    phase = minus_two_pi_over_C*phase*frequency

    # No GPU implementation of exp yet
    #return tf.exp(tf.complex(0.0, phase), name='complex_phase')
    return tf.complex(tf.cos(phase), tf.sin(phase))


def complex_phase_numpy(lm, uvw, frequency):
    nsrc, _ = lm.shape
    ntime, na, _ = uvw.shape
    nchan, = frequency.shape

    lm = lm.reshape(nsrc, 1, 1, 1, 2)
    uvw = uvw.reshape(1, ntime, na, 1, 3)
    frequency = frequency.reshape(1, 1, 1, nchan)

    l, m = lm[:, :, :, :, 0], lm[:, :, :, :, 1]
    u, v, w = uvw[:, :, :, :, 0], uvw[:, :, :, :, 1], uvw[:, :, :, :, 2]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    real_phase = -2*np.pi*1j*(l*u + m*v + n*w)*frequency/lightspeed
    return np.exp(real_phase)


class TestComplexPhase(unittest.TestCase):
    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                         if d.device_type == 'GPU']

    def test_complex_phase(self):
        type_permutations = [
            [np.float32, np.complex64],
            [np.float64, np.complex128]]

        for FT, CT in type_permutations:
            self._impl_test_complex_phase(FT, CT)

    def _impl_test_complex_phase(self, FT, CT):
        nsrc, ntime, na, nchan = 100, 50, 64, 128

        # Set up our numpy input arrays
        np_lm = np.random.random(size=(nsrc, 2)).astype(FT)*0.1
        np_uvw = np.random.random(size=(ntime, na, 3)).astype(FT)
        np_frequency = np.linspace(1.3e9, 1.5e9, nchan,
                                   endpoint=True, dtype=FT)

        np_args = [np_lm, np_uvw, np_frequency]
        tf_names = ['lm', 'uvw', 'frequency']

        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, tf_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.phase(*tf_args, CT=CT)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.compat.v1.global_variables_initializer()

        # Now create a tensorflow Session to evaluate the above
        with tf.compat.v1.Session() as S:
            S.run(init_op)

            phase_np = complex_phase_numpy(np_lm, np_uvw, np_frequency)
            phase_cpu = S.run(cpu_op)
            assert np.allclose(phase_cpu, phase_np)

            for phase_gpu in S.run(gpu_ops):
                assert np.allclose(phase_cpu, phase_gpu)
