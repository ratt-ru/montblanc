import os

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
from montblanc.impl.rime.tensorflow import load_tf_lib
rime = load_tf_lib()

dtype = np.float64
ctype = np.complex64 if dtype == np.float32 else np.complex128
rf = lambda *s: np.random.random(size=s).astype(dtype)
rc = lambda *s: rf(*s) + rf(*s)*1j

nsrc, ntime, na, nchan, npol = 10, 20, 7, 16, 4

np_bsqrt = rc(nsrc, ntime, nchan, npol)
np_complex_phase = rc(nsrc, ntime, na, nchan);
np_feed_rotation = rc(ntime, na, npol)
np_ejones = rc(nsrc, ntime, na, nchan, npol)

args = map(lambda v, n: tf.Variable(v, name=n),
    [np_bsqrt, np_complex_phase, np_feed_rotation, np_ejones],
    ["bsqrt", "complex_phase", "feed_rotation", "ejones"])

def ekb_sqrt(bsqrt, complex_phase, feed_rotation, ejones):
    from montblanc.impl.rime.v4.cpu.CPUSolver import CPUSolver
    result = bsqrt[:,:,np.newaxis,:,:]*complex_phase[:,:,:,:,np.newaxis]

    fr_shape = feed_rotation.shape[0:-1] + (2,2)
    res_shape = result.shape[0:-1] + (2,2)

    # time, antenna, i, j
    # src, time, antenna, channel, j, k
    result = np.einsum("taij,stacjk->stacik",
            feed_rotation.reshape(fr_shape),
            result.reshape(res_shape))

    result = CPUSolver.jones_multiply(ejones, result)
    return result.reshape(nsrc, ntime, na, nchan, npol)

# Pin the compute to the CPU
with tf.device('/cpu:0'):
    expr_cpu = rime.ekb_sqrt(*args, FT=dtype)

# Pin the compute to the GPU
with tf.device('/gpu:0'):
    expr_gpu = rime.ekb_sqrt(*args, FT=dtype)

init_op = tf.global_variables_initializer()

with tf.Session() as S:
    S.run(init_op)

    # Run our expressions on CPU and GPU
    result_cpu = S.run(expr_cpu)
    result_gpu = S.run(expr_gpu)
    result_np = ekb_sqrt(np_bsqrt, np_complex_phase, np_feed_rotation, np_ejones)

    # Check that CPU and GPU results agree
    assert result_cpu.shape == result_gpu.shape
    assert np.allclose(result_cpu, result_np)
    assert np.allclose(result_cpu, result_gpu)

