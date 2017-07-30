import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

def brightness_numpy(stokes, alpha, frequency, ref_freq, pol_type):
    nsrc, ntime, _ = stokes.shape
    nchan, = frequency.shape

    I = stokes[:,:,0].reshape(nsrc, ntime, 1)
    Q = stokes[:,:,1].reshape(nsrc, ntime, 1)
    U = stokes[:,:,2].reshape(nsrc, ntime, 1)
    V = stokes[:,:,3].reshape(nsrc, ntime, 1)

    if pol_type == "linear":
        pass
    elif pol_type == "circular":
        Q, U, V = V, Q, U
    else:
        raise ValueError("Invalid pol_type '{}'".format(pol_type))

    # Compute the spectral index
    freq_ratio = frequency[None,None,:]/ref_freq[:,None,None]
    power = np.power(freq_ratio, alpha[:,:,None])

    CT = np.complex128 if stokes.dtype == np.float64 else np.complex64

    # Compute the brightness matrix
    B = np.empty(shape=(nsrc, ntime, nchan, 4), dtype=CT)
    B[:,:,:,0] = power*(I+Q)
    B[:,:,:,1] = power*(U+V*1j)
    B[:,:,:,2] = power*(U-V*1j)
    B[:,:,:,3] = power*(I-Q)

    return B

class TestBSqrt(unittest.TestCase):
    """ Tests the BSqrt operator """

    def setUp(self):
        # Load the rime operation library
        from montblanc.impl.rime.tensorflow import load_tf_lib
        self.rime = load_tf_lib()
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name for d in device_lib.list_local_devices()
                                if d.device_type == 'GPU']

    def test_b_sqrt(self):
        """ Test the BSqrt operator """
        # List of type constraint for testing this operator
        permutations = [[np.float32, np.complex64, 'linear', {'rtol': 1e-3}],
                        [np.float32, np.complex64, 'circular', {'rtol': 1e-3}],
                        [np.float64, np.complex128, 'linear', {}],
                        [np.float64, np.complex128, 'circular', {}]]

        # Run test with the type combinations above
        for FT, CT, pol_type, tols in permutations:
            self._impl_test_b_sqrt(FT, CT, pol_type, tols)

    def _impl_test_b_sqrt(self, FT, CT, pol_type, tols):
        """ Implementation of the BSqrt operator test """

        nsrc, ntime, na, nchan = 10, 50, 27, 32

        # Useful random floats functor
        rf = lambda *s: np.random.random(size=s).astype(FT)
        rc = lambda *s: (rf(*s) + 1j*rf(*s)).astype(CT)

        # Set up our numpy input arrays

        # Stokes parameters, should produce a positive definite matrix
        stokes = np.empty(shape=(nsrc, ntime, 4), dtype=FT)
        Q = stokes[:,:,1] = rf(nsrc, ntime) - 0.5
        U = stokes[:,:,2] = rf(nsrc, ntime) - 0.5
        V = stokes[:,:,3] = rf(nsrc, ntime) - 0.5
        noise = rf(nsrc, ntime)*0.1
        # Need I^2 = Q^2 + U^2 + V^2 + noise^2
        stokes[:,:,0] = np.sqrt(Q**2 + U**2 + V**2 + noise)

        # Choose random flux to invert
        mask = np.random.randint(0, 2, size=(nsrc, ntime)) == 1
        stokes[mask,0] = -stokes[mask,0]

        # Make the last matrix zero to test the positive semi-definite case
        stokes[-1,-1,:] = 0

        alpha = rf(nsrc, ntime)*0.8
        frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=FT)
        ref_freq = 0.2e9*rf(nsrc,) + 1.3e9

        # Argument list
        np_args = [stokes, alpha, frequency, ref_freq]
        # Argument string name list
        arg_names = ["stokes", "alpha", "frequency", "ref_freq"]

        # Constructor tensorflow variables
        tf_args = [tf.Variable(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return self.rime.b_sqrt(*tf_args, CT=CT,
                                        polarisation_type=pol_type)

        # Pin operation to CPU
        cpu_op = _pin_op('/cpu:0', *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        # Initialise variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as S:
            S.run(init_op)

            # Get the CPU bsqrt and invert flag
            cpu_bsqrt, cpu_invert = S.run(cpu_op)

            # Get our actual brightness matrices
            b = brightness_numpy(stokes, alpha, frequency, ref_freq, pol_type)
            b_2x2 = b.reshape(nsrc, ntime, nchan, 2, 2)
            b_sqrt_2x2 = cpu_bsqrt.reshape(nsrc, ntime, nchan, 2, 2)

            # Multiplying the square root matrix
            # by it's hermitian transpose
            square = np.einsum("...ij,...kj->...ik",
                b_sqrt_2x2, b_sqrt_2x2.conj())

            # Apply any sign inversions
            square[:,:,:,:,:] *= cpu_invert[:,:,None,None,None]

            # And we should obtain the brightness matrix
            assert np.allclose(b_2x2, square)

            # Compare with GPU bsqrt and invert flag
            for gpu_bsqrt, gpu_invert in S.run(gpu_ops):
                self.assertTrue(np.all(cpu_invert == gpu_invert))

                # Compare cpu and gpu
                d = np.isclose(cpu_bsqrt, gpu_bsqrt, **tols)
                d = np.invert(d)
                p = np.nonzero(d)

                if p[0].size == 0:
                    continue

                import itertools
                it = (np.asarray(p).T, cpu_bsqrt[d], gpu_bsqrt[d])
                it = enumerate(itertools.izip(*it))

                msg = ["%s %s %s %s %s" % (i, idx, c, g, c-g)
                            for i, (idx, c, g) in it]

                self.fail("CPU/GPU bsqrt failed likely because the "
                        "last polarisation for each entry differs slightly "
                        "for FT=np.float32 and CT=np.complex64. "
                        "FT='%s' CT='%s'."
                        "Here are the values\n%s" % (FT, CT, '\n'.join(msg)))


if __name__ == "__main__":
    unittest.main()



