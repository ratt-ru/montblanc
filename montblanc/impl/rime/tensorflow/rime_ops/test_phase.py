import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from montblanc.impl.rime.tensorflow.tensorflow_ops import phase as phase_op

lightspeed = 299792458.

def get_dim_indexes(uvw):
    dims = len(uvw.shape) - 1

    all_ = slice(None)

    lm_idx = (all_,) + (None,)*dims + (None,)
    uvw_idx = (None,) + (all_,)*dims + (None,)
    chan_idx =(None,)* dims + (all_,)

    return lm_idx, uvw_idx, chan_idx

def complex_phase_numpy(lm, uvw, frequency):
    """ Compute complex phase using numpy """

    # Set up slicing depending on whether a row based uvw
    # scheme is used
    dims = uvw.ndim - 1
    all_ = slice(None)

    lm_idx, uvw_idx, _ = get_dim_indexes(uvw)

    l = lm[lm_idx + (0,)]
    m = lm[lm_idx + (1,)]

    u = uvw[uvw_idx + (0,)]
    v = uvw[uvw_idx + (1,)]
    w = uvw[uvw_idx + (2,)]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    real_phase = -2*np.pi*1j*(l*u + m*v + n*w)*frequency/lightspeed
    return np.exp(real_phase)

def complex_phase_tf(lm, uvw, frequency, dtype=None):
    """
    Compute the complex phase from lm, uvw and frequency Tensors
    """

    if dtype is None:
        dtype = lm.dtype

    # Get the dynamic shape of input tensors
    lm_shape = tf.shape(lm)
    uvw_shape = tf.shape(uvw)
    frequency_shape = tf.shape(frequency)

    # The shapes are themselves tensors
    nsrc = lm_shape[0]
    ntime, na = uvw_shape[0], uvw_shape[1]
    nchan = frequency_shape[0]

    # Define some constants
    one = tf.constant(1.0, dtype=dtype)
    minus_two_pi_over_C = tf.constant(-2.0*np.pi/lightspeed, dtype=dtype)

    # Reshape now so that we get broadcasting in later operations
    # Need to pack list since list contains tensors, e.g. nsrc
    dims = len(uvw.shape) - 1
    all_ = slice(None)

    lm_idx, uvw_idx, chan_idx = get_dim_indexes(uvw)

    l = lm[lm_idx + (0,)]
    m = lm[lm_idx + (1,)]

    u = uvw[uvw_idx + (0,)]
    v = uvw[uvw_idx + (1,)]
    w = uvw[uvw_idx + (2,)]

    frequency = frequency[chan_idx]

    n = tf.sqrt(one - l**2 - m**2) - one

    # Outer product l*u + m*v * n*w
    phase = l*u + m*v +n*w

    # Multiply in constants
    phase = minus_two_pi_over_C*phase*frequency

    # No GPU implementation of exp yet
    return tf.complex(tf.cos(phase), tf.sin(phase))

class TestComplexPhase(unittest.TestCase):
    """ Tests the ComplexPhase operator """

    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']
    def test_complex_phase(self):
        """ Test the ComplexPhase operator """

        # List of type constraints for testing this operator
        type_permutations = [[np.float32, np.complex64],
                            [np.float64, np.complex128]]

        perms = [[type_permutations[0], True],
                 [type_permutations[1], True],
                 [type_permutations[0], False],
                 [type_permutations[1], False]]

        for (FT, CT), use_row in perms:
            self._impl_test_complex_phase(FT, CT, use_row)
    def _impl_test_complex_phase(self, FT, CT, use_row):
        """ Implementation of the ComplexPhase operator test """

        nsrc, ntime, na, nchan = 10, 15, 16, 16

        uvw_shape = (ntime*na,3) if use_row else (ntime,na,3)

        # Set up our numpy input arrays
        lm = np.random.random(size=(nsrc,2)).astype(FT)*0.1
        uvw = np.random.random(size=uvw_shape).astype(FT)
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
        cpu_expr = _pin_op('/cpu:0', complex_phase_tf, *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, phase_op, *tf_args, CT=CT)
                                for d in self.gpu_devs]
        gpu_exprs = [_pin_op(d, complex_phase_tf, *tf_args)
                                for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU complex phase
            cpu_cplx_phase = S.run(cpu_op)
            tf_cplx_phase = S.run(cpu_expr)

            # Compare vs tensorflow
            self.assertTrue(np.allclose(cpu_cplx_phase, tf_cplx_phase))

            # Compare vs numpy
            np_cplx_phase = complex_phase_numpy(lm, uvw, frequency)
            self.assertTrue(np.allclose(np_cplx_phase, cpu_cplx_phase))

            # Compare vs GPU
            for gpu_op, gpu_expr in zip(gpu_ops, gpu_exprs):
                gpu_cplx_phase, gpu_cp_expr = S.run([gpu_op, gpu_expr])

                self.assertTrue(np.allclose(cpu_cplx_phase, gpu_cplx_phase))
                self.assertTrue(np.allclose(cpu_cplx_phase, gpu_cp_expr))

if __name__ == "__main__":
    unittest.main()
