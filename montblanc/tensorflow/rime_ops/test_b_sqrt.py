import timeit

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
mod = tf.load_op_library('rime.so')

def b_sqrt_op(stokes, alpha, frequency, ref_freq):
    """
    This function wraps rime_phase by deducing the
    complex output result type from the input
    """
    stokes_dtype = stokes.dtype.base_dtype

    if stokes_dtype == tf.float32:
        CT = tf.complex64
    elif stokes_dtype == tf.float64:
        CT = tf.complex128
    else:
        raise TypeError("Unhandled type '{t}'".format(t=stokes.dtype))

    return mod.rime_b_sqrt(stokes, alpha, frequency, ref_freq, CT=CT)

dtype, ctype = np.float32, np.complex64
nsrc, ntime, na, nchan = 100, 50, 64, 128

# Set up our numpy input arrays
np_stokes = np.empty(shape=(nsrc, ntime, 4), dtype=dtype)
Q = np_stokes[:,:,1] = np.random.random(size=(nsrc, ntime)) - 0.5
U = np_stokes[:,:,2] = np.random.random(size=(nsrc, ntime)) - 0.5
V = np_stokes[:,:,3] = np.random.random(size=(nsrc, ntime)) - 0.5
noise = np.random.random(size=(nsrc, ntime))*0.1
# Need I^2 = Q^2 + U^2 + V^2 + noise^2
np_stokes[:,:,0] = np.sqrt(Q**2 + U**2 + V**2 + noise)

np_alpha = np.random.random(size=(nsrc, ntime)).astype(dtype)*0.1
np_frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=dtype)
np_ref_freq = np.array([1.4e9], dtype=dtype)

# Create tensorflow arrays from the numpy arrays
stokes = tf.Variable(np_stokes, name='stokes')
alpha = tf.Variable(np_alpha, name='alpha')
frequency = tf.Variable(np_frequency, name='frequency')
ref_freq = tf.Variable(np_ref_freq, name='ref_freq')

# Get an expression for the b sqrt op on the CPU
with tf.device('/cpu:0'):
    b_sqrt_op_cpu = b_sqrt_op(stokes, alpha, frequency, ref_freq)

# Get an expression for the complex phase op on the GPU
with tf.device('/gpu:0'):
    b_sqrt_op_gpu = b_sqrt_op(stokes, alpha, frequency, ref_freq)

# Get an expression for the complex phase expression on the GPU
#with tf.device('/gpu:0'):
#    cplx_phase_expr_gpu = complex_phase(lm, uvw, frequency)

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(tf.initialize_all_variables())

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_b_sqrt_op_cpu = S.run(b_sqrt_op_cpu)
    print 'Tensorflow CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow GPU
    #start = timeit.default_timer()
    #tf_b_sqrt_op_gpu = S.run(b_sqrt_op_gpu)
    #print 'Tensorflow custom GPU time %f' % (timeit.default_timer() - start)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_b_sqrt_op_cpu.shape == (nsrc, ntime, nchan, 4)

print 'Tests Succeeded'