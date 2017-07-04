
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

class TestEBeam(unittest.TestCase):
    """ Tests the EBeam operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_e_beam(self):
        """ Test the EBeam operator """
        # List of type constraint for testing this operator
        type_permutations = [[np.float32, np.complex64],
                             [np.float64, np.complex128]]

        # Run test with the type combinations above
        for FT, CT in type_permutations:
            self._impl_test_e_beam(FT, CT)

    def _impl_test_e_beam(self, FT, CT):
        """ Implementation of the EBeam operator test """

        nsrc, ntime, na, nchan = 20, 29, 14, 64
        beam_lw = beam_mh = beam_nud = 50

        # Useful random floats functor
        rf = lambda *s: np.random.random(size=s).astype(FT)
        rc = lambda *s: (rf(*s) + 1j*rf(*s)).astype(CT)

        # Set up our numpy input arrays
        lm = (rf(nsrc, 2) - 0.5) * 1e-1
        frequency = np.linspace(1e9, 2e9, nchan,dtype=FT)
        point_errors = (rf(ntime, na, nchan, 2) - 0.5) * 1e-2
        antenna_scaling = rf(na, nchan, 2)
        parallactic_angle = np.deg2rad(rf(ntime, na))
        parallactic_angle_sin = np.sin(parallactic_angle)
        parallactic_angle_cos = np.cos(parallactic_angle)
        beam_extents = FT([-0.9, -0.8, 1e9, 0.8, 0.9, 2e9])
        beam_freq_map = np.linspace(1e9, 2e9, beam_nud, dtype=FT, endpoint=True)
        e_beam = rc(beam_lw, beam_mh, beam_nud, 4)

        # Argument list
        np_args = [lm, frequency, point_errors, antenna_scaling,
                     parallactic_angle_sin, parallactic_angle_cos,
                     beam_extents, beam_freq_map, e_beam]
        # Argument string name list
        arg_names = ["lm", "frequency", "point_errors", "antenna_scaling",
                     "parallactic_angle_sin", "parallactic_angle_cos",
                     "beam_extents", "beam_freq_map", "e_beam"]

        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.e_beam(*tf_args)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU ejones
            cpu_ejones = S.run(cpu_op)

            # Check that most of them are non-zero
            nz_cpu = np.count_nonzero((cpu_ejones))
            self.assertTrue(nz_cpu > 0.8*cpu_ejones.size,
                "Less than 80% of the ejones turns are non-zero")

            # Compare with GPU ejones
            for gpu_ejones in S.run(gpu_ops):
                self.assertTrue(np.allclose(cpu_ejones, gpu_ejones),
                    "This may fail due to discrepancies when rounding "
                    "floating point values near an integer value. "
                    "As the CPU and GPU may slightly differ, "
                    "values may be slightly below and integer on the CPU "
                    "and slghtly above on the GPU for instance. "
                    "It may be appropriate to ignore this assert check. ")

                proportion_acceptable = 1e-4
                d = np.invert(np.isclose(cpu_ejones, gpu_ejones))
                incorrect = d.sum()
                proportion_incorrect = incorrect / float(d.size)

                self.assertTrue(proportion_incorrect < proportion_acceptable,
                    "Proportion of incorrect E beam values {pi} "
                    "({i} out of {t}) is greater than the "
                    "accepted tolerance {pa}.".format(
                        pi=proportion_incorrect, i=incorrect,
                        t=d.size, pa=proportion_acceptable))

if __name__ == "__main__":
    unittest.main()