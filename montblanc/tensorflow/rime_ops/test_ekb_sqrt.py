import numpy as np
import tensorflow as tf

# Load the shared library with the operation
rime = tf.load_op_library('rime.so')

dtype = np.float64
ctype = np.complex64 if dtype == np.float32 else np.complex128
rf = lambda *s: np.random.random(size=s).astype(dtype)
rc = lambda *s: rf(*s) + rf(*s)*1j

nsrc, ntime, na, nchan, npol = 10, 20, 7, 16, 4

np_complex_phase = rc(nsrc, ntime, na, nchan);
np_bsqrt = rc(nsrc, na, nchan, npol)
np_ejones = rc(nsrc, ntime, na, nchan, npol)

args = map(lambda v, n: tf.Variable(v, name=n),
    [np_complex_phase, np_bsqrt, np_ejones],
    ["complex_phase", "bsqrt", "ejones"])

# Pin the compute to the CPU
with tf.device('/cpu:0'):
    expr_cpu = rime.ekb_sqrt(*args, FT=dtype)

# Pin the compute to the GPU
with tf.device('/gpu:0'):
    expr_gpu = rime.ekb_sqrt(*args, FT=dtype)

with tf.Session() as S:
    S.run(tf.initialize_all_variables())

    # Run our expressions on CPU and GPU
    result_cpu = S.run(expr_cpu)
    result_gpu = S.run(expr_gpu)

    # Check that CPU and GPU results agree
    assert result_cpu.shape == result_gpu.shape
    assert np.allclose(result_cpu, result_gpu)

