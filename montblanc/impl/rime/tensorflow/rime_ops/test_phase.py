import os
import timeit

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
from montblanc.impl.rime.tensorflow import load_tf_lib
rime = load_tf_lib()

def complex_phase_op(lm, uvw, frequency):
    """
    This function wraps rime_phase by deducing the
    complex output result type from the input
    """
    lm_dtype = lm.dtype.base_dtype

    if lm_dtype == tf.float32:
        CT = tf.complex64
    elif lm_dtype == tf.float64:
        CT = tf.complex128
    else:
        raise TypeError("Unhandled type '{t}'".format(t=lm.dtype))

    return rime.phase(lm, uvw, frequency, CT=CT)

def complex_phase(lm, uvw, frequency):
    """
    Compute the complex phase from lm, uvw and frequency expressions
    """

    # Get the dynamic shape of input tensors
    lm_shape = tf.shape(lm)
    uvw_shape = tf.shape(uvw)
    frequency_shape = tf.shape(frequency)

    # The shapes are themselves tensors
    nsrc = lm_shape[0]
    ntime, na = uvw_shape[0], uvw_shape[1]
    nchan = frequency_shape[0]

    # Define some constants
    one = tf.constant(1.0, dtype=dtype)
    minus_two_pi_over_C = tf.constant(-2.0*np.pi/lightspeed, dtype=dtype)

    # Reshape now so that we get broadcasting in later operations
    # Need to pack list since list contains tensors, e.g. nsrc
    l = tf.reshape(lm[:,0], tf.pack([nsrc,1,1,1]))
    m = tf.reshape(lm[:,1], tf.pack([nsrc,1,1,1]))

    u = tf.reshape(uvw[:,:,0], tf.pack([1,ntime,na,1]))
    v = tf.reshape(uvw[:,:,1], tf.pack([1,ntime,na,1]))
    w = tf.reshape(uvw[:,:,2], tf.pack([1,ntime,na,1]))

    frequency = tf.reshape(frequency, tf.pack([1,1,1,nchan]))

    n = tf.sqrt(one - l**2 - m**2) - one

    # Outer product l*u + m*v * n*w
    phase = tf.convert_to_tensor(l*u + m*v + n*w, name='real_phase')

    # Multiply in constants
    phase = minus_two_pi_over_C*phase*frequency

    # No GPU implementation of exp yet
    #return tf.exp(tf.complex(0.0, phase), name='complex_phase')
    return tf.complex(tf.cos(phase), tf.sin(phase))

def complex_phase_numpy(lm, uvw, frequency):
    nsrc, _ = lm.shape
    ntime, na, _ = uvw.shape
    nchan, = frequency.shape

    lm = lm.reshape(nsrc, 1, 1, 1, 2)
    uvw = uvw.reshape(1, ntime, na, 1, 3)
    frequency = frequency.reshape(1, 1, 1, nchan)

    l, m = lm[:,:,:,:,0], lm[:,:,:,:,1]
    u, v, w = uvw[:,:,:,:,0], uvw[:,:,:,:,1], uvw[:,:,:,:,2]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    real_phase = -2*np.pi*1j*(l*u + m*v + n*w)*frequency/lightspeed
    return np.exp(real_phase)

dtype, ctype = np.float64, np.complex128
nsrc, ntime, na, nchan = 100, 50, 64, 128
lightspeed = 299792458.

# Set up our numpy input arrays
np_lm = np.random.random(size=(nsrc,2)).astype(dtype)*0.1
np_uvw = np.random.random(size=(ntime,na,3)).astype(dtype)
np_frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=dtype)

# Create tensorflow arrays from the numpy arrays
lm = tf.Variable(np_lm, name='lm')
uvw = tf.Variable(np_uvw, name='uvw')
frequency = tf.Variable(np_frequency, name='frequency')
#lm, uvw, frequency = map(tf.Variable, [np_lm, np_uvw, np_frequency])

# Get an expression for the complex phase op on the CPU
with tf.device('/cpu:0'):
    cplx_phase_op_cpu = complex_phase_op(lm, uvw, frequency)

# Get an expression for the complex phase op on the GPU
with tf.device('/gpu:0'):
    cplx_phase_op_gpu = complex_phase_op(lm, uvw, frequency)

# Get an expression for the complex phase expression on the GPU
with tf.device('/gpu:0'):
    cplx_phase_expr_gpu = complex_phase(lm, uvw, frequency)

init_op = tf.global_variables_initializer()

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(init_op)

    # Evaluate and time tensorflow GPU
    start = timeit.default_timer()
    tf_cplx_phase_op_gpu = S.run(cplx_phase_op_gpu)
    print 'Tensorflow custom GPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow GPU
    start = timeit.default_timer()
    tf_cplx_phase_expr_gpu = S.run(cplx_phase_expr_gpu)
    print 'Tensorflow expression GPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_cplx_phase_op_cpu = S.run(cplx_phase_op_cpu)
    print 'Tensorflow CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time numpy CPU
    start = timeit.default_timer()
    # Now calculate the complex phase using numpy
    # Reshapes help us to broadcast
    np_cplx_phase = complex_phase_numpy(np_lm, np_uvw, np_frequency)
    print 'Numpy CPU time %f' % (timeit.default_timer() - start)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_cplx_phase_op_cpu.shape == (nsrc, ntime, na, nchan)
    assert tf_cplx_phase_op_gpu.shape == (nsrc, ntime, na, nchan)
    assert tf_cplx_phase_expr_gpu.shape == (nsrc, ntime, na, nchan)
    assert np_cplx_phase.shape == (nsrc, ntime, na, nchan)
    assert np.allclose(tf_cplx_phase_op_cpu, np_cplx_phase)
    assert np.allclose(tf_cplx_phase_op_gpu, np_cplx_phase)
    assert np.allclose(tf_cplx_phase_expr_gpu, np_cplx_phase)

print 'Tests Succeeded'