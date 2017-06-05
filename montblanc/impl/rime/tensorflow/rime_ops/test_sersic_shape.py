import os

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
from montblanc.impl.rime.tensorflow import load_tf_lib
rime = load_tf_lib()

dtype = np.float32
ngsrc, ntime, na, nchan = 10, 15, 7, 16
nbl = na*(na-1)//2

rf = lambda *s: np.random.random(size=s).astype(dtype=dtype)

np_uvw = rf(ntime, na, 3)
np_ant1, np_ant2 = map(lambda x: np.int32(x), np.triu_indices(na, 1))
np_ant1, np_ant2 = (np.tile(np_ant1, ntime).reshape(ntime, nbl),
    np.tile(np_ant2, ntime).reshape(ntime,nbl))
np_frequency = np.linspace(1.4e9, 1.5e9, nchan).astype(dtype)
np_sersic_params = rf(3, ngsrc)*np.array([1.0,1.0,np.pi/648000],dtype=dtype)[:,np.newaxis]

assert np_ant1.shape == (ntime, nbl), np_ant1.shape
assert np_ant2.shape == (ntime, nbl), np_ant2.shape
assert np_frequency.shape == (nchan,)

args = map(lambda v, n: tf.Variable(v, name=n),
    [np_uvw, np_ant1, np_ant2, np_frequency, np_sersic_params],
    ["uvw", "ant1", "ant2", "frequency", "sersic_params"])

with tf.device('/cpu:0'):
    sersic_shape_cpu = rime.sersic_shape(*args)

with tf.device('/gpu:0'):
    sersic_shape_gpu = rime.sersic_shape(*args)

init_op = tf.global_variables_initializer()

with tf.Session() as S:
    S.run(init_op)
    tf_sersic_shape_gpu = S.run(sersic_shape_gpu)
    tf_sersic_shape_cpu = S.run(sersic_shape_cpu)
    assert np.allclose(tf_sersic_shape_cpu, tf_sersic_shape_gpu)


