from attrdict import AttrDict
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

# Load the shared library with the operation
rime = tf.load_op_library('rime.so')

dtype = np.float64
np_apply_dies = np.bool(True)

# Infer our complex type
ctype = np.complex64 if dtype == np.float32 else np.complex128

# Lambdas for creating random floats and complex numbers
rf = lambda *s: np.random.random(size=s).astype(dtype)
rc = lambda *s: rf(*s) + rf(*s)*1j

# Problem dimensions
ntime, na, nchan = 25, 64, 64
src_counts = npsrc, ngsrc, nssrc = 5, 6, 7
src_types = POINT, GAUSS, SERSIC = 'point', 'gauss', 'sersic'
nsrc = sum(src_counts)
src_chunk = 5
nbl = na*(na-1)//2
npol = 4
beam_lw = beam_mh = beam_nud = 100

D = AttrDict()

# uvw coordinates
D.uvw = rf(ntime, na, 3)

# frequency and referency frequency
D.frequency = np.linspace(1.4e9, 1.6e9, nchan, dtype=dtype)
D.ref_freq = np.array([1.5e9], dtype=dtype)

# Pointing errors, scaled around (0,0)
D.point_errors = (rf(ntime, na, nchan, 2) - 0.5)*0.01
D.antenna_scaling = rf(na, nchan, 2)
D.ebeam = rc(beam_lw, beam_mh, beam_nud, npol)
D.parallactic_angle = dtype(np.deg2rad(1))
D.beam_ll, D.beam_lm, D.beam_ul, D.beam_um = dtype(
    [-1, -1, 1, 1])

# Antenna pairs
D.ant1, D.ant2 = map(lambda x: np.int32(x), np.triu_indices(na, 1))
D.ant1, D.ant2 = (np.tile(D.ant1, ntime).reshape(ntime, nbl),
                    np.tile(D.ant2, ntime).reshape(ntime, nbl))

# Flags, g term and model visibilities
D.flag = np.random.randint(0, 1,
    size=(ntime, nbl, nchan, npol), dtype=np.uint8)
D.gterm = rc(ntime, na, nchan, npol)

# Create tensorflow variables from numpy arrays
args = AttrDict((n,tf.Variable(v, name=n)) for n, v in D.iteritems())

def get_src_vars(nsrc, src_type=None):
    """ Get lm, stokes, alpha and shape arrays for given src_type """
    src_type = src_type or POINT

    # lm coordinates
    lm = (rf(nsrc, 2) - 0.5)*0.1

    # Stokes parameters
    # Need I**2 = Q**2 + U**2 + V**2 + noise**2
    stokes = rf(nsrc, ntime, npol)
    Q, U, V = stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
    stokes[:,:,0] = Q**2 + U**2 + V**2 + (rf(nsrc, ntime)*0.1)**2

    # Alpha
    alpha = rf(nsrc, ntime)*0.8

    # Shape parameters
    if src_type == POINT:
        param = None
    elif src_type == GAUSS:
        # Shape parameters
        param = rf(3, nsrc)*dtype([0.1,0.1,1])[:,np.newaxis]
    elif src_type == SERSIC:
        param = rf(3, nsrc)*dtype([1,1,np.pi/648000])[:,np.newaxis]
    else:
        raise ValueError("Unknown source type {t}.".format(t=src_type))

    return lm, stokes, alpha, param

# Create some model visibilities
model_vis = tf.Variable(rc(ntime, nbl, nchan, npol), name='model_vis')

base = 0
for src_count, src_type in zip(src_counts, src_types):
    # Generate some source arrays for this source type
    np_lm, np_stokes, np_alpha, np_param = get_src_vars(src_count, src_type)

    # Loop to src_count in src_chunk chunks
    for i in range(0, src_count, src_chunk):
        chunk_end = min(i+src_chunk, src_count)

        # Create tensorflow variables from nump array slices
        lm = tf.Variable(np_lm[i:chunk_end,:])
        stokes = tf.Variable(np_stokes[i:chunk_end])
        alpha = tf.Variable(np_alpha[i:chunk_end])

        # Compute the complex phase
        cplx_phase = rime.phase(lm, args.uvw,
            args.frequency, CT=ctype)

            # Compute the brightness square root
        bsqrt = rime.b_sqrt(stokes, alpha, args.frequency,
            args.ref_freq, CT=ctype)

        # Compute the ejones from the beam cube
        ejones = rime.e_beam(lm, args.point_errors,
            args.antenna_scaling, args.ebeam, args.parallactic_angle,
            args.beam_ll, args.beam_lm,
            args.beam_ul, args.beam_um)
            
        # Compute per antenna jones terms    
        ant_jones = rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=dtype)

        # Compute shape parameters
        if src_type == POINT:
            shape = tf.ones([chunk_end-i, ntime, nbl, nchan], dtype=dtype)
        elif src_type == GAUSS:
            gauss_param = tf.Variable(np_param[:,i:chunk_end])
            shape = rime.gauss_shape(args.uvw, args.ant1, args.ant2,
                args.frequency, gauss_param,
                name='{t}_shape'.format(t=GAUSS))
        elif src_type == SERSIC:
            sersic_param = tf.Variable(np_param[:,i:chunk_end])
            shape = rime.sersic_shape(args.uvw, args.ant1, args.ant2,
                args.frequency, sersic_param,
                name='{t}_shape'.format(t=SERSIC))


        # Sum coherencies over this source type
        model_vis = rime.sum_coherencies(args.ant1, args.ant2, shape,
                ant_jones, args.flag, args.gterm, model_vis, False)

    # Increase base offset by source count
    base += src_count

with tf.Session() as S:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()
    S.run(tf.initialize_all_variables())
    
    for _ in range(10):
        vis = S.run(model_vis, options=run_options, run_metadata=run_metadata)

    print vis.shape

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)


