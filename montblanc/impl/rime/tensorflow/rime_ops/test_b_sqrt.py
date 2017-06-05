import os
import timeit

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
from montblanc.impl.rime.tensorflow import load_tf_lib
rime = load_tf_lib()

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

    return rime.b_sqrt(stokes, alpha, frequency, ref_freq, CT=CT)

def brightness_numpy(stokes, alpha, frequency, ref_freq):
    nsrc, ntime, _ = stokes.shape
    nchan, = frequency.shape

    I = stokes[:,:,0].reshape(nsrc, ntime, 1)
    Q = stokes[:,:,1].reshape(nsrc, ntime, 1)
    U = stokes[:,:,2].reshape(nsrc, ntime, 1)
    V = stokes[:,:,3].reshape(nsrc, ntime, 1)

    # Compute the spectral index
    freq_ratio = (frequency/ref_freq)[np.newaxis,np.newaxis,:]
    power = np.power(freq_ratio, alpha[:,:,np.newaxis])

    # Compute the brightness matrix
    B = np.empty(shape=(nsrc, ntime, nchan, 4), dtype=ctype)
    B[:,:,:,0] = power*(I+Q)
    B[:,:,:,1] = power*(U+V*1j)
    B[:,:,:,2] = power*(U-V*1j)
    B[:,:,:,3] = power*(I-Q)

    return B

dtype, ctype = np.float64, np.complex128
nsrc, ntime, na, nchan = 10, 50, 27, 32

# Set up our numpy input arrays

# Stokes parameters, should produce a positive definite matrix
np_stokes = np.empty(shape=(nsrc, ntime, 4), dtype=dtype)
Q = np_stokes[:,:,1] = np.random.random(size=(nsrc, ntime)) - 0.5
U = np_stokes[:,:,2] = np.random.random(size=(nsrc, ntime)) - 0.5
V = np_stokes[:,:,3] = np.random.random(size=(nsrc, ntime)) - 0.5
noise = np.random.random(size=(nsrc, ntime))*0.1
# Need I^2 = Q^2 + U^2 + V^2 + noise^2
np_stokes[:,:,0] = np.sqrt(Q**2 + U**2 + V**2 + noise)

# Choose random flux to invert
mask = np.random.randint(0, 2, size=(nsrc, ntime)) == 1
np_stokes[mask,0] = -np_stokes[mask,0]

# Make the last matrix zero to test the positive semi-definite case
np_stokes[-1,-1,:] = 0

np_alpha = np.random.random(size=(nsrc, ntime)).astype(dtype)*0.8
np_frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=dtype)
np_ref_freq = np.full_like(np_frequency, 1.4e9)

# Create tensorflow arrays from the numpy arrays
stokes = tf.Variable(np_stokes, name='stokes')
alpha = tf.Variable(np_alpha, name='alpha')
frequency = tf.Variable(np_frequency, name='frequency')
ref_freq = tf.Variable(np_ref_freq, name='ref_freq')

# Get an expression for the b sqrt op on the CPU
with tf.device('/cpu:0'):
    b_sqrt_op_cpu, invert_op_cpu = b_sqrt_op(stokes, alpha, frequency, ref_freq)

# Get an expression for the complex phase op on the GPU
with tf.device('/gpu:0'):
    b_sqrt_op_gpu, invert_op_gpu = b_sqrt_op(stokes, alpha, frequency, ref_freq)

init_op = tf.global_variables_initializer()

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(init_op)

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_b_sqrt_op_cpu, invert_cpu = S.run([b_sqrt_op_cpu, invert_op_cpu])
    print 'Tensorflow CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow GPU
    start = timeit.default_timer()
    tf_b_sqrt_op_gpu, invert_gpu = S.run([b_sqrt_op_gpu, invert_op_gpu])
    print 'Tensorflow GPU time %f' % (timeit.default_timer() - start)

    # Get our actual brightness matrices
    np_b = brightness_numpy(np_stokes, np_alpha, np_frequency, np_ref_freq)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_b_sqrt_op_cpu.shape == (nsrc, ntime, nchan, 4)
    assert np.allclose(tf_b_sqrt_op_cpu, tf_b_sqrt_op_gpu)
    assert np.all(invert_cpu == invert_gpu)

    # Reshape for 2x2 jones multiply below
    b_flat = np_b.reshape(nsrc, ntime, nchan, 2, 2)
    b_sqrt_flat = tf_b_sqrt_op_cpu.reshape(nsrc, ntime, nchan, 2, 2)

    # Multiplying the square root matrix
    # by it's hermitian transpose
    square = np.einsum("...ij,...kj->...ik",
        b_sqrt_flat, b_sqrt_flat.conj())

    # Apply any sign inversions
    square[:,:,:,:,:] *= invert_cpu[:,:,np.newaxis,np.newaxis,np.newaxis]

    # And we should obtain the brightness matrix
    assert np.allclose(b_flat, square)

print 'Tests Succeeded'