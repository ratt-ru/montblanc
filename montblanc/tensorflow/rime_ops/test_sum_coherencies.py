import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
mod = tf.load_op_library('rime.so')

def sum_coherencies_op(*args):
    uvw_dtype = args[0].dtype.base_dtype

    if uvw_dtype == tf.float32:
        CT = tf.complex64
    elif uvw_dtype == tf.float64:
        CT = tf.complex128
    else:
        raise TypeError("Unhandled type '{t}'".format(t=lm.dtype))

    return mod.rime_sum_coherencies(*args)

ntime, na, nchan = 20, 7, 32
nbl = na*(na-1)//2
dtype = np.float32
npsrc, ngsrc, nssrc = 20, 20, 20
nsrc = npsrc+ngsrc+nssrc

rf = lambda *s: np.random.random(size=s).astype(dtype)

np_uvw = rf(ntime, na, 3)
np_gauss_shape = rf(ngsrc, 3)
np_sersic_shape = rf(nssrc, 3)
np_frequency = rf(nchan)
np_ant1, np_ant2 = map(lambda x: np.int32(x), np.triu_indices(na, 1))
np_ant1, np_ant2 = np.tile(np_ant1, ntime), np.tile(np_ant2, ntime)
np_ant_jones = rf(nsrc, ntime, na, nchan, 4) + rf(nsrc, ntime, na, nchan, 4)*1j
np_flag = np.zeros(shape=(ntime, nbl, nchan, 4)).astype(np.uint8)
np_weight = rf(ntime, nbl, nchan, 4)
np_g_term = rf(ntime, na, nchan, 4) + rf(ntime, na, nchan, 4)*1j
np_obs_vis = rf(ntime, nbl, nchan, 4) + rf(ntime, nbl, nchan, 4)*1j

args = map(lambda n, s: tf.Variable(n, name=s),
    [np_uvw, np_gauss_shape, np_sersic_shape,
    np_frequency, np_ant1, np_ant2, np_ant_jones,
    np_flag, np_weight, np_g_term, np_obs_vis],
    ["uvw", "gauss_shape", "sersic_shape",
    "frequency", "ant1", "ant2", "ant_jones",
    "flag", "weight", "g_term", "observed_vis"])

with tf.device('/cpu:0'):
    sum_coh_op_cpu = sum_coherencies_op(*args)

with tf.Session() as S:
    S.run(tf.initialize_all_variables())
    tf_sum_coh_op_cpu = S.run(sum_coh_op_cpu)