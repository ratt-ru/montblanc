import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from montblanc.impl.rime.tensorflow.tensorflow_ops import (
                    sum_coherencies as sum_coherencies_op)

class TestSumCoherencies(unittest.TestCase):
    """ Tests the SumCoherencies operator """

    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_sum_coherencies(self):
        """ Test the SumCoherencies operator """

        # List of type constraints for testing this operator
        type_permutations = [[[np.float32, np.complex64], {'rtol': 1e-4}],
                             [[np.float64, np.complex128], {}]]

        # Permute the complex phase on and off
        perms = []
        for type_perms in type_permutations:
            perms.append(type_perms + [True])
            perms.append(type_perms + [False])

        # Run test with the type combinations above
        for (FT, CT), cmp_kw, cplx_phase in perms:
            self._impl_test_sum_coherencies(FT, CT, cmp_kw, cplx_phase)

    def _impl_test_sum_coherencies(self, FT, CT, cmp_kw, have_complex_phase):
        """ Implementation of the SumCoherencies operator test """

        rf = lambda *a, **kw: np.random.random(*a, **kw).astype(FT)
        rc = lambda *a, **kw: rf(*a, **kw) + 1j*rf(*a, **kw).astype(CT)

        from montblanc.impl.rime.tensorflow.rime_ops.op_test_utils import random_baselines

        nsrc, ntime, na, nchan = 10, 15, 7, 16
        nbl = na*(na-1)//2

        chunks = np.random.random_integers(int(3.*nbl/4.), nbl, ntime)
        nvrow = np.sum(chunks)

        _, np_ant1, np_ant2, np_time_index = random_baselines(chunks, na)

        np_shape = rf(size=(nsrc, nvrow, nchan))
        np_ant_jones = rc(size=(nsrc, ntime, na, nchan, 4))
        np_sgn_brightness = np.random.randint(0, 3, size=(nsrc, ntime), dtype=np.int8) - 1
        np_complex_phase = rc(size=(nsrc,nvrow,nchan))
        np_base_coherencies = rc(size=(nvrow, nchan, 4))

        # Argument list
        np_args = [np_time_index, np_ant1, np_ant2, np_shape, np_ant_jones,
            np_sgn_brightness, np_complex_phase, np_base_coherencies]
        # Argument string name list
        arg_names = ['time_index', 'antenna1', 'antenna2', 'shape', 'ant_jones',
            'sgn_brightness', 'complex_phase', 'base_coherencies']
        is_list = [False, False, False, False, False, True, True, True]
        # Constructor tensorflow variables
        tf_args = [[tf.Variable(v, name=n)] if l else tf.Variable(v, name=n)
                    for v, n, l
                    in zip(np_args, arg_names, is_list)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return sum_coherencies_op(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU coherencies
            cpu_coh = S.run(cpu_op)

            # Compare against the GPU coherencies
            for gpu_coh in S.run(gpu_ops):
                if not np.allclose(cpu_coh, gpu_coh, **cmp_kw):
                    if FT == np.float32:
                        self.fail("CPU and GPU results don't match for "
                                  "single precision float data. Consider "
                                  "relaxing the tolerance")
                    else:
                        self.fail("CPU and GPU results don't match!")


if __name__ == "__main__":
    unittest.main()

