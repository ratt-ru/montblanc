from attrdict import AttrDict
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

# Load the shared library with the operation
rime = tf.load_op_library('rime.so')

# Set our floating point precision and infer the complex type
dtype = np.float64
ctype = np.complex64 if dtype == np.float32 else np.complex128

# Lambdas for creating random floats and complex numbers
rf = lambda *s: np.random.random(size=s).astype(dtype)
rc = lambda *s: rf(*s) + rf(*s)*1j

# Problem dimensions
ntime, na, nchan = 25, 16, 16
src_counts = [5, 6, 7]
tf_src_counts = [tf.Variable(v) for v in src_counts]
src_types = POINT, GAUSS, SERSIC = ['point', 'gauss', 'sersic']
src_chunk = tf.constant(5)
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
    lm = (tf.random_normal([nsrc, 2], dtype=dtype,
        name='lm_{t}'.format(t=src_type)) - 0.5)*0.1

    # Stokes parameters
    # Need I**2 = Q**2 + U**2 + V**2 + noise**2
    Q = tf.random_normal([nsrc, ntime, 1], dtype=dtype) - 0.5
    U = tf.random_normal([nsrc, ntime, 1], dtype=dtype) - 0.5
    V = tf.random_normal([nsrc, ntime, 1], dtype=dtype) - 0.5
    noise = tf.random_normal([nsrc, ntime, 1], dtype=dtype)*0.1
    I = Q**2 + U**2 + V**2 + noise**2
    stokes = tf.concat(2, [I, Q, U, V], name='stokes')

    # Alpha
    alpha = tf.random_normal([nsrc, ntime], dtype=dtype, name='alpha')*0.8

    # Shape parameters
    if src_type == POINT:
        shape = tf.ones([nsrc, ntime, nbl, nchan],
            dtype=dtype, name='{t}_shape'.format(t=POINT))
    elif src_type == GAUSS:
        # Shape parameters
        gauss_factor = tf.constant([0.1, 0.1, 1.0], dtype=dtype)
        gauss_param = (tf.random_normal([3, nsrc], dtype=dtype)*
            tf.reshape(gauss_factor, [3, 1]))

        shape = rime.gauss_shape(args.uvw, args.ant1, args.ant2,
            args.frequency, gauss_param,
            name='{t}_shape'.format(t=GAUSS))
    elif src_type == SERSIC:
        sersic_factor = tf.constant([1,1,np.pi/648000], dtype=dtype)
        sersic_param = (tf.random_normal([3, nsrc], dtype=dtype)*
            tf.reshape(sersic_factor, [3,1]))

        shape = rime.sersic_shape(args.uvw, args.ant1, args.ant2,
            args.frequency, sersic_param,
            name='{t}_shape'.format(t=SERSIC))
    else:
        raise ValueError("Unknown source type {t}.".format(t=src_type))

    return lm, stokes, alpha, shape


# Create some model visibilities
model_vis = tf.Variable(rc(ntime, nbl, nchan, npol), name='model_vis')

nsrc = tf.reduce_sum(tf_src_counts)
base = tf.Variable(0)

for src_index, src_type in enumerate(src_types):
    src_count = tf_src_counts[src_index]
    # Obtain source arrays for this source type
    lm, stokes, alpha, shape = get_src_vars(src_count, src_type)
    
    def cond(i, base, model_vis):
        return tf.less(i, src_count)

    def body(i, base, model_vis):
        chunk_end = tf.minimum(i + src_chunk, src_count)
        chunk_size = chunk_end - i
        base_end = tf.minimum(base + src_chunk, nsrc)
        apply_dies = tf.equal(base_end, nsrc)

        lm_slice     = tf.slice(lm,     [i, 0      ], [chunk_size, -1        ])
        stokes_slice = tf.slice(stokes, [i, 0, 0   ], [chunk_size, -1, -1    ])
        alpha_slice  = tf.slice(alpha,  [i, 0      ], [chunk_size, -1        ])
        shape_slice  = tf.slice(shape,  [i, 0, 0, 0], [chunk_size, -1, -1, -1])

        # Compute the complex phase
        with tf.control_dependencies([lm_slice]):
            cplx_phase = rime.phase(lm_slice, args.uvw,
                args.frequency, CT=ctype, name='complex_phase')

            # Compute the brightness square root
        with tf.control_dependencies([cplx_phase]):
            bsqrt = rime.b_sqrt(stokes_slice, alpha_slice, args.frequency,
                args.ref_freq, CT=ctype, name='b_sqrt')

        # Compute the ejones from the beam cube
        with tf.control_dependencies([bsqrt]):
            ejones = rime.e_beam(lm_slice, args.point_errors,
                args.antenna_scaling, args.ebeam, args.parallactic_angle,
                args.beam_ll, args.beam_lm,
                args.beam_ul, args.beam_um,
                name='e_beam')
            
        # Compute per antenna jones terms    
        with tf.control_dependencies([ejones]):
            ant_jones = rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=dtype,
                name='ekb_sqrt')

        # Sum coherencies over this source type
        with tf.control_dependencies([ant_jones]):
            model_vis = rime.sum_coherencies(args.ant1, args.ant2, shape_slice,
                    ant_jones, args.flag, args.gterm, model_vis, apply_dies,
                    name='sum_coherencies')

        return [chunk_end, base_end, model_vis]

    _, base, model_vis = tf.while_loop(cond, body, [tf.Variable(0), base, model_vis],
        parallel_iterations=1)

with tf.Session() as S:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_options = tf.RunOptions()
    run_metadata = tf.RunMetadata()
    S.run(tf.initialize_all_variables())
    
    for _ in range(10):
        vis, b, ns = S.run([model_vis, base, nsrc],
            options=run_options, run_metadata=run_metadata)

    print vis.shape, b, ns

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)


