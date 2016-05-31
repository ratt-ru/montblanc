import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
mod = tf.load_op_library('rime.so')

def sum_coherencies_op(uvw, obs_vis):
    uvw_dtype = uvw.dtype.base_dtype

    if uvw_dtype == tf.float32:
        CT = tf.complex64
    elif uvw_dtype == tf.float64:
        CT = tf.complex128
    else:
        raise TypeError("Unhandled type '{t}'".format(t=lm.dtype))

    return mod.rime_sum_coherencies(uvw, obs_vis)

ntime, na, nchan = 20, 7, 32
nbl = na*(na-1)//2
dtype = np.float32

rf = lambda *s: np.random.random(size=s).astype(dtype)

np_uvw = rf(ntime, na, 3)
np_obs_vis = rf(ntime, nbl, nchan, 4) + rf(ntime, nbl, nchan, 4)*1j

args = map(lambda n, s: tf.Variable(n, name=s),
    [np_uvw, np_obs_vis],
    ["uvw", "obs_vis"])

with tf.device('/cpu:0'):
    sum_coh_op_cpu = sum_coherencies_op(*args)

with tf.Session() as S:
    S.run(tf.initialize_all_variables())
    tf_sum_coh_op_cpu = S.run(sum_coh_op_cpu)