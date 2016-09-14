import itertools
import os
import subprocess
import tempfile

import numpy as np
import pyrap.tables as pt

import Tigger
import Tigger.Models.SkyModel as tsm
from Tigger.Models.Formats.AIPSCCFITS import lm_to_radec

#=========================================
# Directory and Script Configuration
#=========================================

# Directory that holds MS and Beam data
data_dir = 'data'

# Directory in which we expect our measurement set to be located
msfile = os.path.join(data_dir, 'WSRT.MS')
model_data_column = 'MODEL_DATA'

# Directory in which meqtree-related files are read/written
meq_dir = 'meqtrees'
# Scripts
meqpipe = 'meqtree-pipeliner.py'
# Meqtree profile and script
cfg_file = os.path.join(meq_dir, 'tdlconf.profiles')
sim_script = os.path.join(meq_dir, 'turbo-sim.py')
tigger_sky_file = os.path.join(meq_dir, 'sky_model.txt')

# Directory in which we expect our beams to be located
beam_dir = os.path.join(data_dir, 'beams')
beam_file_prefix = 'beam'
base_beam_file = os.path.join(beam_dir, beam_file_prefix)
beam_file_pattern = ''.join((base_beam_file, '_$(corr)_$(reim).fits'))

# Find the location of the meqtree pipeliner script
meqpipe_actual = subprocess.check_output(['which', meqpipe]).strip()
cfg_section = 'montblanc-compare'

#=========================================
# Source Configuration
#=========================================

source_coords = [[25,10],[30,5],[8,6],[23,7],[15,10]]
IQUVs =  [[1.0,0,0,0], [2,0.5,0,0], [1.0,0,0,0], [1.0,0,0,0],[1.0,0,0,0]]
alphas = [0.8, 0.7, 0.1, 0.2, 0.3]
#source_coords = np.rad2deg([[0.0008, 0.0036]])
#source_coords = [[0.0, 0.0] for i in xrange(5)]
nsrc = len(source_coords)

pt_lm = np.deg2rad(source_coords)
#pt_stokes = np.tile([[1.,0.,0.,0.]], [nsrc, 1])
pt_stokes = np.asarray(IQUVs)
pt_alpha = np.asarray(alphas)

ref_freq = 1.415e9

assert pt_lm.shape == (nsrc, 2)
assert pt_stokes.shape == (nsrc, 4)
assert pt_alpha.shape == (nsrc,)

print pt_lm.shape
print pt_stokes.shape
print pt_alpha.shape

#=========================================
# Create Tigger LSM
#=========================================

# Need the phase centre for lm_to_radec
with pt.table(msfile + '::FIELD', ack=False) as F:
    ra0, dec0 = F.getcol('PHASE_DIR')[0][0]

# Create the tigger sky model
with open(tigger_sky_file, 'w') as f:
    it = enumerate(itertools.izip(pt_lm, pt_stokes, pt_alpha))
    for i, ((l, m), (I, Q, U, V), alpha) in it:
        ra, dec = lm_to_radec(l, m, ra0, dec0)
        l, m = np.rad2deg([ra,dec])

        f.write('#format: ra_d dec_d i q u v spi freq0\n')
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
    # Measurement Set
    'ms_sel.msname={ms}'.format(ms=msfile),
    # Tigger sky file
    'tiggerlsm.filename={sm}'.format(sm=tigger_sky_file),
    # Output column
    'ms_sel.output_column={c}'.format(c=model_data_column),
    # Imaging Column
    'img_sel.imaging_column={c}'.format(c=model_data_column),
    # Beam FITS file pattern
    'pybeams_fits.filename_pattern={p}'.format(p=beam_file_pattern),
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
from hypercube.dims import Dimension

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
        return [Dimension('npsrc', pt_alpha.shape[0])]

slvr_cfg = montblanc.rime_solver_cfg(
    mem_budget=1024*1024*1024,
    data_source=Options.DATA_SOURCE_DEFAULT,
    dtype='double',
    auto_correlations=False,
    version='tf')

slvr = montblanc.rime_solver(slvr_cfg)

ms_mgr = MeasurementSetManager(msfile, slvr, slvr_cfg)

source_providers = [MSSourceProvider(ms_mgr),
    FitsBeamSourceProvider(beam_file_pattern),
    RadioSourceProvider()]

sink_providers = [MSSinkProvider(ms_mgr, 'CORRECTED_DATA')]
slvr.solve(source_providers=source_providers,
    sink_providers=sink_providers)

for obj in source_providers + sink_providers + [ms_mgr]:
    obj.close()

# Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
subprocess.call(cmd_list)


