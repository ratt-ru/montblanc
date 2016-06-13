from attrdict import AttrDict
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

# Load the shared library with the operation
rime = tf.load_op_library('rime.so')

dtype = np.float64
np_apply_dies = np.bool(True)

ctype = np.complex64 if dtype == np.float32 else np.complex128
rf = lambda *s: np.random.random(size=s).astype(dtype)
rc = lambda *s: rf(*s) + rf(*s)*1j

ntime, na, nchan = 15, 7, 16
src_counts = npsrc, ngsrc, nssrc = 5, 6, 7
nsrc = sum(src_counts)
nbl = na*(na-1)//2
npol = 4
beam_lw = beam_mh = beam_nud = 50

D = AttrDict()

# lm coordinates scaled around (0,0)
D.lm = (rf(nsrc, 2) - 0.5)*0.1
# uvw coordinates
D.uvw = rf(ntime, na, 3)

# frequency and referency frequency
D.frequency = np.linspace(1.4e9, 1.6e9, nchan, dtype=dtype)
D.ref_freq = np.array([1.5e9], dtype=dtype)

# Stokes and alpha
D.stokes = rf(nsrc, ntime, npol)
Q, U, V = D.stokes[:,:,1], D.stokes[:,:,2], D.stokes[:,:,3]
# Need I**2 = Q**2 + U**2 + V**2 + noise**2
D.stokes[:,:,0] = Q**2 + U**2 + V**2 + (rf(nsrc, ntime)*0.1)**2
D.alpha = rf(nsrc, ntime)*0.8

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
    size=(ntime, nbl, nchan, 4), dtype=np.uint8)
D.gterm = rc(ntime, na, nchan, npol)

# Shape parameters
D.gauss_param = rf(3, ngsrc)*dtype([0.1,0.1,1])[:,np.newaxis]
D.sersic_param = rf(3, nssrc)*dtype([1,1,np.pi/648000])[:,np.newaxis]

# Create tensorflow variables from numpy arrays
args = AttrDict((n,tf.Variable(v, name=n)) for n, v in D.iteritems())

# Compute the complex phase
cplx_phase = rime.phase(args.lm, args.uvw,
    args.frequency, CT=ctype)

# Compute the brightness square root
bsqrt = rime.b_sqrt(args.stokes, args.alpha, args.frequency,
    args.ref_freq, CT=ctype)

# Compute the ejones from the beam cube
ejones = rime.e_beam(args.lm, args.point_errors,
    args.antenna_scaling, args.ebeam, args.parallactic_angle,
    args.beam_ll, args.beam_lm,
    args.beam_ul, args.beam_um)
    
# Compute per antenna jones terms    
ant_jones = rime.ekb_sqrt(cplx_phase, bsqrt, ejones, FT=dtype)

# Compute the gaussian shape
gauss_shape = rime.gauss_shape(args.uvw, args.ant1, args.ant2,
    args.frequency, args.gauss_param)

# Compute the sersic shape
sersic_shape = rime.sersic_shape(args.uvw, args.ant1, args.ant2,
    args.frequency, args.sersic_param)

# Zero the model visibilities
model_vis = tf.zeros((ntime, nbl, nchan, npol), dtype=ctype)

# Pointing source shapes are one
point_shape = tf.ones((npsrc, ntime, na, nchan), dtype=dtype)

# Slice the antenna jones array, referencing the point sources
point_ant_jones = tf.slice(ant_jones,
    (0,0,0,0,0), (npsrc,-1,-1,-1,-1))

# Sum coherencies over point sources
model_vis = rime.sum_coherencies(args.ant1, args.ant2, point_shape,
        point_ant_jones, args.flag, args.gterm, model_vis, False)

# Slice the antenna jones array, referencing the gaussian sources
gauss_ant_jones = tf.slice(ant_jones,
    (npsrc,0,0,0,0), (ngsrc,-1,-1,-1,-1))

# Sum coherencies over gaussian sources
model_vis = rime.sum_coherencies(args.ant1, args.ant2, gauss_shape,
        gauss_ant_jones, args.flag, args.gterm, model_vis, False)

# Slice the antenna jones array, referencing the sersic sources
sersic_ant_jones = tf.slice(ant_jones,
    (npsrc+ngsrc,0,0,0,0), (nssrc,-1,-1,-1,-1))

# Sum coherencies over sersic sources
model_vis = rime.sum_coherencies(args.ant1, args.ant2, sersic_shape,
        sersic_ant_jones, args.flag, args.gterm, model_vis, True)

with tf.Session() as S:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    S.run(tf.initialize_all_variables())
    
    vis = S.run(model_vis, options=run_options, run_metadata=run_metadata)

    print vis.shape

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)


