#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import logging
import unittest
import subprocess
import os
import sys
import time

import montblanc
import numpy as np

from montblanc.config import (
    RimeSolverConfig as Options)

# Directory in which we expect our measurement set to be located
data_dir = 'data'
msfile = os.path.join(data_dir, 'WSRT.MS')

# Directory in which meqtree-related files are read/written
meq_dir = 'meqtrees'
# Meqtree profile and script
cfg_file = os.path.join(meq_dir, 'tdlconf.profiles')
sim_script = os.path.join(meq_dir, 'turbo-sim.py')

# Tigger conversion scripts
tigger_convert = 'tigger-convert'
meqpipe = 'meqtree-pipeliner.py'
cfg_section = 'montblanc-compare'

# MS column in which to dump data
model_data_column = 'MODEL_DATA'

# Based on http://stackoverflow.com/a/377028/1611416
def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


# This function taken from
# https://github.com/ska-sa/meqtrees-cattery/blob/891783b41fbb3036e4b8a56c3d67ee7b162d4343/Cattery/Meow/Direction.py#L52-L64
def lm_to_radec (l,m,ra0,dec0):
  """ Returns ra,dec corresponding to l,m w.r.t. direction ra0,dec0 """
  # see formula at http://en.wikipedia.org/wiki/Orthographic_projection_(cartography)
  rho = np.sqrt(l**2+m**2)
  if rho == 0.0:
    ra, dec = ra0, dec0
  else:
    cc = np.arcsin(rho);
    ra = ra0 + np.arctan2(l*np.sin(cc),rho*np.cos(dec0)*np.cos(cc)-m*np.sin(dec0)*np.sin(cc))
    dec = np.arcsin(np.cos(cc)*np.sin(dec0) + m*np.sin(cc)*np.cos(dec0)/rho)

  return ra,dec

# Radio source parameters. 1Jy source offset from phase centre
_L = 0.0008
_M = 0.0036
_I = 1.0
_Q = 0.0
_U = 0.0
_V = 0.0
_ALPHA = 0.5
_REF_FREQ = 1.5e9

class TestCmpVis(unittest.TestCase):
    """
    Compares visibilities produced by different versions of Montblanc, and
    with those produced by MeqTrees. This tests the brightness and phase
    matrices associated with a single point sources
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)

        # Add a handler that outputs INFO level logging to file
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)

        montblanc.log.setLevel(logging.INFO)
        montblanc.log.handlers = [fh, ch]

    def tearDown(self):
        """ Tear down each test case """
        pass


    def get_meq_vis(self, msfile):
        import pyrap.tables as pt

        antfile = '::'.join([msfile, 'ANTENNA'])
        freqfile = '::'.join([msfile, 'SPECTRAL_WINDOW'])

        def create_tigger_sky_model():
            """ Create a tigger sky model """

            # Calculate l and m coordinates in degrees
            # with respect to the phase centre
            with pt.table(msfile + '::FIELD', ack=False) as F:
                ra0, dec0 = F.getcol('PHASE_DIR')[0][0]

            ra, dec = lm_to_radec(_L, _M, ra0, dec0)
            l, m = np.rad2deg([ra,dec])

            # Sky model files
            meq_sky_file = os.path.join(meq_dir, 'sky_model.txt')
            tigger_sky_file = os.path.splitext(meq_sky_file)[0] + '.lsm.html'

            with open(meq_sky_file, 'w') as f:
                f.write('#format: ra_d dec_d i q u v spi freq0\n')
                f.write('{l} {m} {i} {q} {u} {v} {spi} {rf}\n'.format(
                    l=l, m=m, i=_I, q=_Q, u=_U, v=_V, spi=_ALPHA, rf=_REF_FREQ))

            # Convert the sky model to tigger format
            subprocess.call([tigger_convert, '-f', meq_sky_file, tigger_sky_file])

            return tigger_sky_file

        tigger_sky_file = create_tigger_sky_model()

        # Find the location of the meqtree pipeliner script
        meqpipe_actual = subprocess.check_output(['which', meqpipe]).strip()

        # Big fat assumption #1: MeqTrees is installed as a system package
        # So we want to call system python
        pyexe = 'python'
        env = os.environ

        # If we're inside a virtual environment look for system python
        if hasattr(sys, 'real_prefix'):
            # Strip out any virtual environment paths which may interfere
            path = env["PATH"].split(os.pathsep)
            env["PATH"] = os.pathsep.join([p for p in path
                if not p.startswith(sys.prefix)])

            # Big fat assumption #2: python is living in some bin directory
            for root, dirs, names in os.walk(os.path.join(sys.real_prefix, 'bin')):
                if pyexe in names:
                    pyexe = os.path.join(root, pyexe)
                    break

        cmd_list = [pyexe,
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
            sim_script,
            '=simulate'
            ]

        # Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
        subprocess.call(cmd_list, env=env)

        # Pull the visibilities out of the dump column
        with pt.table(msfile, ack=False).query('ANTENNA1 != ANTENNA2') as ms, \
            pt.table(antfile, ack=False) as msant, \
            pt.table(freqfile, ack=False) as msfreq:

            # Work out dimensions for the reshape
            na = msant.nrows()
            nbl = na*(na-1)//2
            nchan = msfreq.getcol('NUM_CHAN')
            ntime = ms.nrows() // nbl
            vis = ms.getcol(model_data_column)
            return vis.reshape(ntime, nbl, nchan, 4)


    def get_v2_output(self, slvr_cfg):
        slvr_cfg[Options.VERSION] = Options.VERSION_TWO

        # Get visibilities from the v2 solver
        slvr_cfg[Options.VERSION] = Options.VERSION_TWO
        with montblanc.rime_solver(slvr_cfg) as slvr:
            # Create and transfer lm to the solver
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            l, m = lm[0,:], lm[1,:]
            l[:] = _L
            m[:] = _M
            slvr.transfer_lm(lm)

            # Create and transfer brightness to the solver
            B = np.empty(shape=slvr.brightness.shape, dtype=slvr.brightness.dtype)
            I, Q, U, V, alpha = B[0,:,:], B[1,:,:], B[2,:,:], B[3,:,:], B[4,:,:]
            I[:] = _I
            Q[:] = _Q
            U[:] = _U
            V[:] = _V
            alpha[:] = _ALPHA
            slvr.transfer_brightness(B)

            # Set the reference wavelength
            slvr.set_ref_wave(montblanc.constants.C / _REF_FREQ)

            slvr.solve()

            return slvr.X2, slvr.retrieve_model_vis().transpose(1,2,3,0)

    def get_v4_output(self, slvr_cfg):
        # Get visibilities from the v4 solver
        slvr_cfg[Options.VERSION] = Options.VERSION_FOUR
        with montblanc.rime_solver(slvr_cfg) as slvr:
            # Create and transfer lm to the solver
            lm = np.empty(shape=slvr.lm.shape, dtype=slvr.lm.dtype)
            l, m = lm[:,0], lm[:,1]
            l[:] = _L
            m[:] = _M
            slvr.transfer_lm(lm)

            # Create and transfer stoke and alpha to the solver
            stokes = np.empty(shape=slvr.stokes.shape, dtype=slvr.stokes.dtype)
            alpha = np.empty(shape=slvr.alpha.shape, dtype=slvr.alpha.dtype)
            I, Q, U, V = stokes[:,:,0], stokes[:,:,1], stokes[:,:,2], stokes[:,:,3]
            I[:] = _I
            Q[:] = _Q
            U[:] = _U
            V[:] = _V
            alpha[:] = _ALPHA
            slvr.transfer_stokes(stokes)
            slvr.transfer_alpha(alpha)

            # Set the reference frequency
            slvr.set_ref_freq(_REF_FREQ)

            slvr.solve()

            return slvr.X2, slvr.retrieve_model_vis()

    def get_v5_output(self, slvr_cfg):
        # Get visibilities from the v5 solver
        slvr_cfg[Options.VERSION] = Options.VERSION_FIVE
        with montblanc.rime_solver(slvr_cfg) as slvr:
            slvr.lm[:,0] = _L
            slvr.lm[:,1] = _M

            slvr.stokes[:,:,0] = _I
            slvr.stokes[:,:,1] = _Q
            slvr.stokes[:,:,2] = _U
            slvr.stokes[:,:,3] = _V
            slvr.alpha[:] = _ALPHA

            slvr.set_ref_freq(_REF_FREQ)

            slvr.solve()

            return slvr.X2, slvr.model_vis.copy()


    def test_cmp_visibilities(self):
        """ Test visibilities produced by montblanc and meqtrees """
        if not os.path.exists(msfile):
            raise unittest.SkipTest("MeasurementSet '{ms}' required "
                "for this test is not present".format(ms=msfile))

        slvr_cfg = montblanc.rime_solver_cfg(
            msfile=msfile,
            sources=montblanc.sources(point=1, gaussian=0, sersic=0),
            dtype='double', version=Options.VERSION_TWO)


        v2_chi, v2_vis = self.get_v2_output(slvr_cfg)
        v4_chi, v4_vis = self.get_v4_output(slvr_cfg)
        v5_chi, v5_vis = self.get_v5_output(slvr_cfg)

        # Test that v2, v4 and v5 model visibilities agree
        self.assertTrue(np.allclose(v2_vis, v4_vis))
        self.assertTrue(np.allclose(v2_chi, v4_chi))
        self.assertTrue(np.allclose(v4_vis, v5_vis))
        self.assertTrue(np.allclose(v4_chi, v5_chi))

        try:
            meq_vis = self.get_meq_vis(slvr_cfg[Options.MS_FILE])
            self.assertTrue(np.allclose(meq_vis, v5_vis))
        except Exception as e:
            montblanc.log.exception("Unable to run MeqTrees for "
                "purposes of comparing model visibilities. "
                "This will not be treated as a test failure")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCmpVis)
    unittest.TextTestRunner(verbosity=2).run(suite)
