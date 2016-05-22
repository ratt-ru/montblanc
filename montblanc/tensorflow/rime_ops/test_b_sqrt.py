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
    nsrc, ntime, _ = stokes.shape
    nchan, = frequency.shape

    I = stokes[:,:,0].reshape(nsrc, ntime, 1)
    Q = stokes[:,:,1].reshape(nsrc, ntime, 1)
    U = stokes[:,:,2].reshape(nsrc, ntime, 1)
    V = stokes[:,:,3].reshape(nsrc, ntime, 1)

    # Compute the spectral index
    freq_ratio = frequency[np.newaxis,np.newaxis,:]/np.asscalar(ref_freq)
    power = np.power(freq_ratio, alpha[:,:,np.newaxis])

    # Compute the brightness matrix
    B = np.empty(shape=(nsrc, ntime, nchan, 4), dtype=ctype)
    B[:,:,:,0] = power*(I+Q)
    B[:,:,:,1] = power*(U+V*1j)
    B[:,:,:,2] = power*(U-V*1j)
    B[:,:,:,3] = power*(I-Q)

    # Compute the trace and determinant.
    trace = 2*I
    det = I**2 - Q**2 - U**2 - V**2

    # Compute values for computing matrix square root
    # setting any 0.0 values of t to 1.0 to avoid nans
    # t == 0.0 (and s == 0.0) implies a zero matrix anyway
    s = np.sqrt(det);
    t = np.sqrt(trace + 2*s);
    t[t == 0.0] = 1.0;

    # Take the sqrt of the power
    power_sqrt = np.sqrt(power)

    # Compute the square root of the brightness matrix
    B_sqrt = np.empty(shape=(nsrc, ntime, nchan, 4), dtype=ctype)
    B_sqrt[:,:,:,0] = power_sqrt*(I+Q+s)/t
    B_sqrt[:,:,:,1] = power_sqrt*(U+V*1j)/t
    B_sqrt[:,:,:,2] = power_sqrt*(U-V*1j)/t
    B_sqrt[:,:,:,3] = power_sqrt*(I-Q+s)/t

    return B, B_sqrt

def b_sqrt(stokes, alpha, frequency, ref_freq):
    stokes_shape = tf.shape(stokes)
    frequency_shape = tf.shape(frequency)

    nsrc, ntime = stokes_shape[0], stokes_shape[1]
    nchan = frequency_shape[0]

    src_shape = tf.pack([nsrc, ntime, 1, 1])
    freq_shape = tf.pack([1, 1, nchan, 1])

    I = tf.reshape(stokes[:,:,0], src_shape)
    Q = tf.reshape(stokes[:,:,1], src_shape)
    U = tf.reshape(stokes[:,:,2], src_shape)
    V = tf.reshape(stokes[:,:,3], src_shape)

    # Compute the spectral index
    frequency = tf.reshape(frequency, freq_shape)
    alpha = tf.reshape(alpha, src_shape)
    power = tf.pow(frequency/ref_freq[0], alpha)

    # Compute the brightness matrix
    XX = tf.complex(power*(I+Q), dtype(0.0))
    XY = tf.complex(power*U    , power*V   )
    YX = tf.complex(power*U    , power*(-V))
    YY = tf.complex(power*(I-Q), dtype(0.0))

    B = tf.concat(3, [XX, XY, YX, YY])

    # Compute the trace and determinant.
    trace = 2.0*I
    det = I**2 - Q**2 - U**2 - V**2

    # Compute values for computing matrix square root
    s = tf.sqrt(det);
    t = tf.sqrt(trace + 2.0*s);
    # To avoid nans/infs, set t = 1.0 if t == 0.0
    # as this implies a zero matrix anyway
    mask = tf.equal(t, dtype(0.0))
    t = tf.select(mask, tf.ones(tf.shape(t), dtype=dtype), t)

    # Compute the square root of the power
    power_sqrt = tf.sqrt(power)

    # Compute the square root of the brightness matrix
    XX = tf.complex(power_sqrt*(I+Q+s)/t, dtype(0.0)       )
    XY = tf.complex(power_sqrt*U/t      , power_sqrt*V/t   )
    YX = tf.complex(power_sqrt*U/t      , power_sqrt*(-V)/t)
    YY = tf.complex(power_sqrt*(I-Q+s)/t, dtype(0.0)       )
    
    B_sqrt = tf.concat(3, [XX, XY, YX, YY])

    return B, B_sqrt

dtype, ctype = np.float64, np.complex128
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

# Get an expression for the complex phase op on the GPU
with tf.device('/gpu:0'):
    b_sqrt_expr_gpu = b_sqrt(stokes, alpha, frequency, ref_freq)

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(tf.initialize_all_variables())

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_b_sqrt_op_cpu = S.run(b_sqrt_op_cpu)
    print 'Tensorflow CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_b_sqrt_op_gpu = S.run(b_sqrt_op_gpu)
    print 'Tensorflow GPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_b_expr_cpu, tf_b_sqrt_expr_gpu = S.run(b_sqrt_expr_gpu)
    print 'Tensorflow expression CPU time %f' % (timeit.default_timer() - start)

    start = timeit.default_timer()
    np_b, np_b_sqrt = b_sqrt_numpy(np_stokes, np_alpha, np_frequency, np_ref_freq)
    print 'Numpy CPU time %f' % (timeit.default_timer() - start)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_b_sqrt_op_cpu.shape == (nsrc, ntime, nchan, 4)
    assert np_b_sqrt.shape == (nsrc, ntime, nchan, 4)
    assert np.allclose(tf_b_sqrt_op_cpu, np_b_sqrt)
    assert np.allclose(tf_b_sqrt_op_gpu, np_b_sqrt, rtol=1e-3)
    assert np.allclose(tf_b_expr_cpu, np_b)
    assert np.allclose(tf_b_sqrt_expr_gpu, np_b_sqrt)

    # Reshape for 2x2 jones multiply below
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