import os
import timeit

import numpy as np
import tensorflow as tf

# Load the library containing the custom operation
rime = tf.load_op_library(os.path.join(os.getcwd(), 'rime.so'))

def e_beam_op(lm, frequency, point_errors, antenna_scaling,
        parallactic_angles, beam_extents, ebeam):
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

    return rime.e_beam(lm, frequency, point_errors, antenna_scaling,
        parallactic_angles, beam_extents, ebeam)

dtype, ctype = np.float64, np.complex128
nsrc, ntime, na, nchan = 20, 29, 14, 64
beam_lw = beam_mh = beam_nud = 50

# Beam cube coordinates

# Useful random floats functor
rf = lambda *s: np.random.random(size=s).astype(dtype)

# Set up our numpy input arrays
np_lm = (rf(nsrc,2)-0.5)*1e-1
np_frequency = np.linspace(1e9, 2e9, nchan).astype(dtype)
np_point_errors = (rf(ntime, na, nchan, 2)-0.5)*1e-2
np_antenna_scaling = rf(na,nchan,2)
np_parallactic_angle = np.deg2rad(rf(ntime, na)).astype(dtype)
np_beam_extents = dtype([-1, -1, 1e9, 1, 1, 2e9])
np_e_beam = (rf(beam_lw, beam_mh, beam_nud, 4) +
        1j*rf(beam_lw, beam_mh, beam_nud, 4)).astype(ctype)

# Create tensorflow variables
args = map(lambda n, s: tf.Variable(n, name=s),
    [np_lm, np_frequency, np_point_errors, np_antenna_scaling,
    np_parallactic_angle, np_beam_extents, np_e_beam],
    ["lm", "frequency", "point_errors", "antenna_scaling",
    "parallactic_angles", "beam_extents", "e_beam"])

# Get an expression for the e beam op on the CPU
with tf.device('/cpu:0'):
    e_beam_op_cpu = e_beam_op(*args)

# Get an expression for the e beam op on the GPU
with tf.device('/gpu:0'):
    e_beam_op_gpu = e_beam_op(*args)

# Now create a tensorflow Session to evaluate the above
with tf.Session() as S:
    S.run(tf.initialize_all_variables())

    # Evaluate and time tensorflow CPU
    start = timeit.default_timer()
    tf_e_beam_op_cpu = S.run(e_beam_op_cpu)
    print 'Tensorflow CPU time %f' % (timeit.default_timer() - start)

    # Evaluate and time tensorflow GPU
    start = timeit.default_timer()
    tf_e_beam_op_gpu = S.run(e_beam_op_gpu)
    print 'Tensorflow GPU time %f' % (timeit.default_timer() - start)

    assert tf_e_beam_op_gpu.shape == tf_e_beam_op_cpu.shape

    proportion_acceptable = 1e-4
    d = np.invert(np.isclose(tf_e_beam_op_cpu, tf_e_beam_op_gpu))
    incorrect = d.sum()
    proportion_incorrect = incorrect / float(d.size)

    assert proportion_incorrect < proportion_acceptable, (
        'Proportion of incorrect E beam values {pi} '
        '({i} out of {t}) '
        'is greater than the accepted tolerance {pa}.').format(
            pi=proportion_incorrect,
            i=incorrect,
            t=d.size,
            pa=proportion_acceptable)
