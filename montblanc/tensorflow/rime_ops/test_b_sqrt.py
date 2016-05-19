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

def b_sqrt_numpy(stokes, alpha, frequency, ref_freq):
    I = stokes[:,:,0]
    Q = stokes[:,:,1]
    U = stokes[:,:,2]
    V = stokes[:,:,3]

    # Compute the spectral index
    freq_ratio = frequency[np.newaxis,np.newaxis,:]/np.asscalar(ref_freq)
    power = np.power(freq_ratio, alpha[:,:,np.newaxis])

    B = np.empty(shape=(nsrc, ntime, nchan, 4), dtype=ctype)
    B[:,:,:,0] = (I+Q)[:,:,np.newaxis]*power
    B[:,:,:,1] = (U+V*1j)[:,:,np.newaxis]*power
    B[:,:,:,2] = (U-V*1j)[:,:,np.newaxis]*power
    B[:,:,:,3] = (I-Q)[:,:,np.newaxis]*power

    # Compute the trace and determinant. Need power**2
    # for det since its composed of squares
    trace = (2*I)[:,:,np.newaxis]*power
    det = (I**2 - Q**2 - U**2 - V**2)[:,:,np.newaxis]*(power**2)

    # Compute values for computing matrix square root
    s = np.sqrt(det);
    t = np.sqrt(trace + 2.0*s);

    B_sqrt = B.copy()

    # Add s to the diagonals
    B_sqrt[:,:,:,0] += s
    B_sqrt[:,:,:,3] += s

    # Divide all matrix entries by t
    t[t == 0.0] = 1.0;
    B_sqrt /= t[:,:,:,np.newaxis]

    return B, B_sqrt

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

np_alpha = np.random.random(size=(nsrc, ntime)).astype(dtype)*0.8
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

    start = timeit.default_timer()
    np_b, np_b_sqrt = b_sqrt_numpy(np_stokes, np_alpha, np_frequency, np_ref_freq)
    print 'Numpy CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow GPU
    #start = timeit.default_timer()
    #tf_b_sqrt_op_gpu = S.run(b_sqrt_op_gpu)
    #print 'Tensorflow custom GPU time %f' % (timeit.default_timer() - start)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_b_sqrt_op_cpu.shape == (nsrc, ntime, nchan, 4)
    assert np_b_sqrt.shape == (nsrc, ntime, nchan, 4)
    assert np.allclose(tf_b_sqrt_op_cpu, np_b_sqrt)

    # Check that multiplying the matrix
    np_b_flat = np_b.reshape(-1,2,2)
    np_b_sqrt_flat = np_b_sqrt.reshape(-1,2,2)

    # Check that multiplying the square root matrix
    # by it's hermitian transpose yields the original
    assert np.allclose(np_b_flat,
        np.einsum("...ij,...kj->...ik",
            np_b_sqrt_flat, np_b_sqrt_flat.conj()))
    
    # Check that multiplying the square root matrix
    # by itself yields the original
    assert np.allclose(np_b_flat,
        np.einsum("...ij,...jk->...ik",
            np_b_sqrt_flat, np_b_sqrt_flat))

print 'Tests Succeeded'