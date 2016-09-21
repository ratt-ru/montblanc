import os

import numpy as np
import tensorflow as tf

# Load the shared library with the operation
rime = tf.load_op_library(os.path.join(os.getcwd(), 'rime.so'))

dtype = np.float64
np_apply_dies = np.bool(True)

ctype = np.complex64 if dtype == np.float32 else np.complex128
rf = lambda *s: np.random.random(size=s).astype(dtype)

nsrc, ntime, na, nchan = 10, 15, 7, 16
nbl = na*(na-1)//2

np_ant1, np_ant2 = map(lambda x: np.int32(x), np.triu_indices(na, 1))
np_ant1, np_ant2 = (np.tile(np_ant1, ntime).reshape(ntime, nbl),
    np.tile(np_ant2, ntime).reshape(ntime,nbl))
np_shape = rf(nsrc, ntime, nbl, nchan)
np_ant_jones = rf(nsrc, ntime, na, nchan, 4) + rf(nsrc, ntime, na, nchan, 4)*1j
np_sgn_brightness = np.random.randint(0, 3, size=(nsrc, ntime), dtype=np.int8) - 1
np_flag = np.random.randint(0, 2, size=(ntime, nbl, nchan, 4), dtype=np.uint8)
np_gterm = rf(ntime, na, nchan, 4) + rf(ntime, na, nchan, 4)*1j
np_model_vis = rf(ntime, nbl, nchan, 4) + rf(ntime, nbl, nchan, 4)*1j


args = map(lambda v, n: tf.Variable(v, name=n),
    [np_ant1, np_ant2, np_shape, np_ant_jones, np_sgn_brightness,
    np_flag, np_gterm, np_model_vis, np_apply_dies],
    ["ant1", "ant2", "shape", "ant_jones", "sgn_brightness",
    "flag", "gterm", "model_vis", "apply_dies"])

# Pin the compute to the CPU
with tf.device('/cpu:0'):
    expr_cpu = rime.sum_coherencies(*args)

# Pin the compute to the GPU
with tf.device('/gpu:0'):
    expr_gpu = rime.sum_coherencies(*args)

with tf.Session() as S:
    S.run(tf.initialize_all_variables())

    # Run our expressions on CPU and GPU
    result_cpu = S.run(expr_cpu)
    result_gpu = S.run(expr_gpu)

    assert result_cpu.shape == result_gpu.shape
    assert np.allclose(result_cpu, result_gpu)

