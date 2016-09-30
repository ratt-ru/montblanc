import itertools
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyrap.tables as pt

#=========================================
# Directory and Script Configuration
#=========================================

# Directory that holds MS and Beam data
data_dir = 'data'

# Directory in which we expect our measurement set to be located
msfile = os.path.join(data_dir, 'WSRT.MS')
meq_vis_column = 'MODEL_DATA'
mb_vis_column = 'CORRECTED_DATA'

# Directory in which meqtree-related files are read/written
meq_dir = 'meqtrees'
# Scripts
meqpipe = 'meqtree-pipeliner.py'
# Meqtree profile and script
cfg_file = os.path.join(meq_dir, 'tdlconf.profiles')
sim_script = os.path.join(meq_dir, 'turbo-sim.py')
tigger_sky_file = os.path.join(meq_dir, 'sky_model.txt')

# Directory in which we expect our beams to be located
beam_on = 1
beam_dir = os.path.join(data_dir, 'beams')
beam_file_prefix = 'beam'
base_beam_file = os.path.join(beam_dir, beam_file_prefix)
beam_file_pattern = ''.join((base_beam_file, '_$(corr)_$(reim).fits'))
l_axis = '-X'

# Find the location of the meqtree pipeliner script
meqpipe_actual = subprocess.check_output(['which', meqpipe]).strip()
cfg_section = 'montblanc-compare'

#======================================================
# Configure the beam files with frequencies from the MS
#======================================================

from montblanc.impl.rime.tensorflow.sources.fits_beam_source_provider import (
    _create_filenames, _open_fits_files)

# Extract frequencies from the MS
with pt.table(msfile + '::SPECTRAL_WINDOW', ack=False) as SW:
    frequency = SW.getcol('CHAN_FREQ')[0]
    ref_freq = SW.getcol('REF_FREQUENCY')[0]

bandwidth = frequency[-1] - frequency[0]
filenames = _create_filenames(beam_file_pattern)
files = _open_fits_files(filenames)
fgen = (f for (re, im) in files.itervalues() for f in (re, im))

for f in fgen:
    f[0].header['CRVAL3'] = frequency[0]
    f[0].header['CDELT3'] = bandwidth / (f[0].header['NAXIS3']-1)
    f.close()

#=========================================
# Source Configuration
#=========================================

np.random.seed(0)
nsrc = 1
rf = np.random.random

# Source coordinates between -30 and 30 degrees
source_coords = np.empty(shape=(nsrc, 2), dtype=np.float64)
IQUVs = np.empty(shape=(nsrc, 4), dtype=np.float64)
I, Q, U, V = IQUVs[:,0], IQUVs[:,1], IQUVs[:,2], IQUVs[:,3]
alphas = np.empty(shape=(nsrc,), dtype=np.float64)

source_coords[:] = (rf(size=source_coords.shape) - 0.5)*60
Q[:] = rf(size=Q.shape)*0.1
U[:] = rf(size=U.shape)*0.1
V[:] = rf(size=V.shape)*0.1
I[:] = np.sqrt(Q**2 + U**2 + V**2)

alphas[:] = 2*(np.random.random(size=alphas.size) - 0.5)

pt_lm = np.deg2rad(source_coords)
pt_stokes = np.asarray(IQUVs)
pt_alpha = np.asarray(alphas)

assert pt_lm.shape == (nsrc, 2), pt_lm.shape
assert pt_stokes.shape == (nsrc, 4), pt_stokes.shape
assert pt_alpha.shape == (nsrc,), pt_alpha.shape

print pt_lm.shape
print pt_stokes.shape
print pt_alpha.shape

#=========================================
# Create Tigger ASCII sky model
#=========================================

from Tigger.Models.Formats.AIPSCCFITS import lm_to_radec

# Need the phase centre for lm_to_radec
with pt.table(msfile + '::FIELD', ack=False, readonly=True) as F:
    ra0, dec0 = F.getcol('PHASE_DIR')[0][0]

# Create the tigger sky model
with open(tigger_sky_file, 'w') as f:
    f.write('#format: ra_d dec_d i q u v spi freq0\n')
    it = enumerate(itertools.izip(pt_lm, pt_stokes, pt_alpha))
    for i, ((l, m), (I, Q, U, V), alpha) in it:
        ra, dec = lm_to_radec(l, m, ra0, dec0)
        l, m = np.rad2deg([ra,dec])

        f.write('{l:.20f} {m:.20f} {i} {q} {u} {v} {spi} {rf:.20f}\n'.format(
            l=l, m=m, i=I, q=Q, u=U, v=V, spi=alpha, rf=ref_freq))

#=========================================
# Call MeqTrees
#=========================================

cmd_list = ['python',
    # Meqtree Pipeline script
    meqpipe_actual,
    # Configuration File
    '-c', cfg_file,
    # Configuration section
    '[{section}]'.format(section=cfg_section),
    # Enable the beam?
    'me.e_enable = {e}'.format(e=beam_on),
    # Measurement Set
    'ms_sel.msname={ms}'.format(ms=msfile),
    # Tigger sky file
    'tiggerlsm.filename={sm}'.format(sm=tigger_sky_file),
    # Output column
    'ms_sel.output_column={c}'.format(c=meq_vis_column),
    # Imaging Column
    'img_sel.imaging_column={c}'.format(c=meq_vis_column),
    # Beam FITS file pattern
    'pybeams_fits.filename_pattern={p}'.format(p=beam_file_pattern),
    # FITS L AXIS
    'pybeams_fits.l_axis={l}'.format(l=l_axis),
    sim_script,
    '=simulate'
    ]


import montblanc

from montblanc.config import RimeSolverConfig as Options

from montblanc.impl.rime.tensorflow.ms import MeasurementSetManager
from montblanc.impl.rime.tensorflow.sources import (SourceProvider,
    MSSourceProvider,
    FitsBeamSourceProvider)

from montblanc.impl.rime.tensorflow.sinks import MSSinkProvider

class RadioSourceProvider(SourceProvider):
    def name(self):
        return "RadioSourceProvider"

    def point_lm(self, context):
        lp, up = context.dim_extents('npsrc')
        return pt_lm[lp:up, :]

    def point_stokes(self, context):
        (lp, up), (lt, ut) = context.dim_extents('npsrc', 'ntime')
        return np.tile(pt_stokes[lp:up, np.newaxis, :], [1, ut-lt, 1])

    def point_alpha(self, context):
        (lp, up), (lt, ut) = context.dim_extents('npsrc', 'ntime')
        return np.tile(pt_alpha[lp:up, np.newaxis], [1, ut-lt])

    def ref_frequency(self, context):
        return np.full(context.shape, ref_freq, context.dtype)

    def updated_dimensions(self):
        return [('npsrc', pt_lm.shape[0])]

slvr_cfg = montblanc.rime_solver_cfg(
    mem_budget=1024*1024*1024,
    data_source=Options.DATA_SOURCE_DEFAULT,
    dtype='double',
    auto_correlations=False,
    version='tf')

slvr = montblanc.rime_solver(slvr_cfg)

ms_mgr = MeasurementSetManager(msfile, slvr, slvr_cfg)

source_providers = []
source_providers.append(MSSourceProvider(ms_mgr))

if beam_on == 1:
    beam_prov = FitsBeamSourceProvider(beam_file_pattern,
        l_axis=l_axis, m_axis='Y')
    source_providers.append(beam_prov)

source_providers.append(RadioSourceProvider())

sink_providers = [MSSinkProvider(ms_mgr, mb_vis_column)]
slvr.solve(source_providers=source_providers,
    sink_providers=sink_providers)

for obj in source_providers + sink_providers + [ms_mgr]:
    obj.close()

# Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
subprocess.call(cmd_list)

# Compare MeqTree and Montblanc visibilities
with pt.table(msfile, ack=False, readonly=True) as MS:
    ntime, nbl, nchan = slvr.dim_global_size('ntime', 'nbl', 'nchan')
    shape = (ntime, nbl, nchan, 4)
    meq_vis = MS.getcol(meq_vis_column).reshape(shape)
    mb_vis = MS.getcol(mb_vis_column).reshape(shape)

    # Compare
    close = np.isclose(meq_vis, mb_vis)
    not_close = np.invert(close)
    problems = np.nonzero(not_close)

    # Everything agrees, exit
    if problems[0].size == 0:
        print 'Montblanc and MeqTree visibilities agree'
        sys.exit(1)

    bad_vis_file = 'bad_visibilities.txt'

    # Some visibilities differ, do some analysis
    print ("Montblanc differs from MeqTrees by {nc}/{t} visibilities. "
        "Writing them out to '{bvf}'").format(
        nc=problems[0].size, t=not_close.size, bvf=bad_vis_file)

    abs_diff = np.abs(meq_vis - mb_vis)
    rmsd = np.sqrt(np.sum(abs_diff**2)/abs_diff.size)
    nrmsd = rmsd / (np.max(abs_diff) - np.min(abs_diff))
    print 'RMSD {rmsd} NRMSD {nrmsd}'.format(rmsd=rmsd, nrmsd=nrmsd)

    # Plot a histogram of the difference
    try:
        import matplotlib
        matplotlib.use('pdf')
        import matplotlib.pyplot as plt
    except:
        print "Exception importing matplotlib %s" % sys.exc_info[2]
    else:
        nr_of_bins = 100
        n, bins, patches = plt.hist(abs_diff.flatten(),
            bins=np.logspace(np.log10(1e-10), np.log10(1.0), nr_of_bins))

        plt.gca().set_xscale("log")
        plt.xlabel('Magnitude Difference')
        plt.ylabel('Counts')
        plt.grid(True)

        plt.savefig('histogram.pdf')

    mb_problems = mb_vis[problems]
    meq_problems = meq_vis[problems]
    difference = mb_problems - meq_problems
    amplitude = np.abs(difference)

    # Create an iterator over the first 100 problematic visibilities
    t = (np.asarray(problems).T, mb_problems, meq_problems, difference, amplitude)
    it = enumerate(itertools.izip(*t))
    it = itertools.islice(it, 0, 1000, 1)

    # Write out the problematic visibilities to file
    with open(bad_vis_file, 'w') as f:
        for i, (p, mb, meq, d, amp) in it:
            f.write("{i} {t} Montblanc: {mb} MeqTrees: {meq} "
                "Difference {d} Absolute Difference {ad} \n".format(
                    i=i, t=p, mb=mb, meq=meq, d=d, ad=amp))