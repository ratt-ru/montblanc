import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
mod = tf.load_op_library('rime.so')

def complex_phase(lm, uvw, frequency):
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

    return mod.rime_phase(lm, uvw, frequency, CT=CT)

dtype, ctype = np.float32, np.complex64
nsrc, ntime, na, nchan = 100, 50, 64, 128
lightspeed = 299792458.

# Set up our numpy input arrays
lm_np = np.random.random(size=(nsrc,2)).astype(dtype)*0.1
np_uvw = np.random.random(size=(ntime,na,3)).astype(dtype)
np_frequency = np.linspace(1.3e9, 1.5e9, nchan, endpoint=True, dtype=dtype)

# Create tensorflow arrays from the numpy arrays
lm = tf.Variable(lm_np, name='lm')
uvw = tf.Variable(np_uvw, name='uvw')
frequency = tf.Variable(np_frequency, name='frequency')
#lm, uvw, frequency = map(tf.Variable, [lm_np, np_uvw, np_frequency])

# Get an expression for the complex phase on the CPU
with tf.device('/cpu:0'):
    cplx_phase_cpu = complex_phase(lm, uvw, frequency)

# Get an expression for the complex phase on the GPU
with tf.device('/gpu:0'):
    cplx_phase_gpu = complex_phase(lm, uvw, frequency)

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(tf.initialize_all_variables())
    tf_cplx_phase_cpu = S.run(cplx_phase_cpu)
    tf_cplx_phase_gpu = S.run(cplx_phase_gpu)

    # Now calculate the complex phase using numpy
    # Reshapes help us to broadcast
    lm = lm_np.reshape(nsrc, 1, 1, 1, 2)
    uvw = np_uvw.reshape(1, ntime, na, 1, 3)
    frequency = np_frequency.reshape(1, 1, 1,nchan)

    l, m = lm[:,:,:,:,0], lm[:,:,:,:,1]
    u, v, w = uvw[:,:,:,:,0], uvw[:,:,:,:,1], uvw[:,:,:,:,2]

    n = np.sqrt(1.0 - l**2 - m**2) - 1.0
    phase = -2*np.pi*1j*(l*u + m*v + n*w)*frequency/lightspeed
    np_cplx_phase = np.exp(phase)

    # Check that our shapes and values agree with a certain tolerance
    assert tf_cplx_phase_cpu.shape == (nsrc, ntime, na, nchan)
    assert tf_cplx_phase_gpu.shape == (nsrc, ntime, na, nchan)
    assert np_cplx_phase.shape == (nsrc, ntime, na, nchan)
    assert np.allclose(tf_cplx_phase_cpu, np_cplx_phase)
    assert np.allclose(tf_cplx_phase_gpu, np_cplx_phase)

print 'Tests Succeeded'