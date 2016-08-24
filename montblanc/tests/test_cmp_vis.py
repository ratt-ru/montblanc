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

from collections import OrderedDict
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

# Directory in which we expect our beams to be located
#beam_dir = os.path.join(data_dir, 'beams')
beam_dir = os.path.join(data_dir, 'beams')
beam_file_prefix = 'beam'
base_beam_file = os.path.join(beam_dir, beam_file_prefix)
beam_file_pattern = ''.join((base_beam_file, '_$(xy)_$(reim).fits'))

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
# _L = 0.0000001
# _M = 0.0000001
_I = 1.0
# _Q = 0.2
# _U = 0.25
# _V = 0.3
_Q = 0
_U = 0
_V = 0
# _ALPHA = 0.8
# _REF_FREQ = 1.5e9
_ALPHA = 0.0
_REF_FREQ = 1.415e9

# Cut and paste from meqtrees-cattery InterpolatedBeams.py
class FITSAxes (object):
  """Helper class encapsulating a FITS header."""
  def __init__ (self,hdr):
    """Creates FITSAxes object from FITS header""";
    self._axis = {};
    naxis = hdr['NAXIS'];
    self._naxis = [0]*naxis;
    self._grid = [[]]*naxis;
    self._type = ['']*naxis;
    self._rpix = [0]*naxis;
    self._rval = [0]*naxis;
    self._rval0 = [0]*naxis;
    self._grid = [None]*naxis;
    self._w2p  = [None]*naxis;
    self._p2w  = [None]*naxis;
    self._delta = [1]*naxis;
    self._delta0 = [1]*naxis;
    self._unit = [None]*naxis;
    self._unit_scale = [1.]*naxis;
    # extract per-axis info
    for i in range(naxis):
      ax = str(i+1);
      nx = self._naxis[i] = hdr.get('NAXIS'+ax);
      # CTYPE is axis name
      self._type[i] = ctype = hdr.get('CTYPE'+ax,None);
      if self._type[i] is not None:
        self._axis[self._type[i]] = i;
      # axis gridding
      # use non-standard keywords GRtype1 .. GRtypeN to supply explicit grid values and ignore CRVAL/CRPIX/CDELT
      grid = [ hdr.get('G%s%d'%(ctype,j),None) for j in range(1,nx+1) ];
      if all([x is not None for x in grid]):
        self._grid[i] = numpy.array(grid);
        self._w2p[i] = interpolate.interp1d(grid,range(len(grid)),'linear');
        self._p2w[i] = interpolate.interp1d(range(len(grid)),grid,'linear');
      else:
        self._rval[i] = self._rval0[i] = rval = hdr.get('CRVAL'+ax,0);
        self._rpix[i] = rpix = hdr.get('CRPIX'+ax,1) - 1;
        self._delta[i] = self._delta0[i] = delta = hdr.get('CDELT'+ax,1);
        self._setup_grid(i);
      self._unit[i] = hdr.get('CUNIT'+ax,'').strip().upper();

  def _setup_grid (self,i):
    """Internal helper to set up the grid based on rval/rpix/delta"""
    nx,rpix,rval,delta = self._naxis[i],self._rpix[i],self._rval[i],self._delta[i];
    self._grid[i] = (np.arange(0.,float(nx))-rpix)*delta+rval;
    self._w2p[i] =  lambda world,rpix=rpix,rval=rval,delta=delta:rpix+(world-rval)/delta;
    self._p2w[i] =  lambda pix,rpix=rpix,rval=rval,delta=delta:(pix-rpix)*delta+rval;

  def ndim (self):
    return len(self._naxis);

  def naxis (self,axis):
    return self._naxis[self.iaxis(axis)];

  def iaxis (self,axisname):
    return axisname if isinstance(axisname,int) else self._axis.get(axisname,-1);

  def grid (self,axis):
    return self._grid[self.iaxis(axis)];

  def type (self,axis):
    return self._type[self.iaxis(axis)];

  def unit (self,axis):
    return self._unit[self.iaxis(axis)];

  def setUnitScale (self,axis,scale):
    iaxis = self.iaxis(axis);
    self._unit_scale[iaxis] = scale;
    self._rval[iaxis] = self._rval0[iaxis]*scale;
    self._delta[iaxis] = self._delta0[iaxis]*scale;
    self._setup_grid(iaxis);

  def toPixel (self,axis,world,sign=1):
    """Converts array of world coordinates to pixel coordinates""";
    return self._w2p[self.iaxis(axis)](sign*world);

  def toWorld (self,axis,pixel,sign=1):
    """Converts array of pixel coordinates to world coordinates""";
    return self._p2w[self.iaxis(axis)](pixel)*sign;

def simulate_fits_beam(l, m, freqs):
    img = np.zeros(shape=(l,m), dtype=np.float64)

    lr = np.arange(-l//2, l//2)[:,np.newaxis]
    mr = np.arange(-m//2, m//2)[np.newaxis,:]

    img += np.where(lr**2 + mr**2 > 1800, 1, 0)

class FitsBeam(object):
    def __init__(self, base_filename):
        self._base_filename = base_filename
        self._filenames = self._create_filenames(base_filename)
        self._files = self._open_fits_files(self._filenames)
        self._configure_axes()

    def _create_filenames(self, base_filename):
        correlations = ('xx', 'xy', 'yx', 'yy')
        reim = ('re', 'im')

        def _re_im_filenames(corr, base):
            return tuple('{b}_{c}_{ri}.fits'.format(
                    b=base, c=corr, ri=ri)
                for ri in reim)

        return OrderedDict(
            (c, _re_im_filenames(c, base_filename))
            for c in correlations)

    def _open_fits_files(self, filenames):
        from astropy.io import fits
        fits_open_args = { 'mode' : 'update', 'memmap' : False }

        return OrderedDict(
            (corr, tuple(fits.open(fn, **fits_open_args) for fn in files))
            for corr, files in filenames.iteritems() )        

    def _configure_axes(self):
        """ Configure the axes object """
        re0, im0 = self._files.values()[0]

        # Create a Cattery FITSAxes object
        self._axes = axes = FITSAxes(re0[0].header)

        # Scale any axes in degrees to radians
        for ai in range(axes.ndim()):
            if axes.unit(ai) == 'DEG':
                axes.setUnitScale(ai, np.pi/180.0)

        # Identify the l, m and f axes
        l_ax, m_ax, f_ax = (axes.iaxis(a) for a in ('L', 'M', 'FREQ'))
        l_ax, m_ax = (e if e != -1 else axes.iaxis(a)
            for e, a in zip([l_ax, m_ax], ['X', 'Y']))

        if l_ax == -1 or m_ax == -1:
            raise ValueError("No L or M axes found")

        # Work out the size of each dimension
        lsize, msize, fsize = (axes.naxis(i) for i in (l_ax, m_ax, f_ax))

        # Work out the extents of th beam cube in L, M and FREQ
        beam_ll, beam_ul = axes.toWorld(l_ax, np.array([0,lsize-1]))
        beam_lm, beam_um = axes.toWorld(m_ax, np.array([0,msize-1]))
        beam_lfreq, beam_ufreq = axes.toWorld(f_ax, np.array([0,fsize-1]))

        self._beam_extents = (beam_ll, beam_lm,
            beam_lfreq, beam_ul,
            beam_um, beam_ufreq)

    def reconfigure_frequency_axes(self, frequency):
        """
        Reconfigure the FITS cubes with the given frequency
        array, so that MeqTrees and Montblanc work with
        the same data.
        """
        delta = frequency[-1] - frequency[0]
        for f in (f for re, im in self._files.itervalues() for f in (re, im)):
            f[0].header['CRVAL3'] = frequency[0]
            f[0].header['CDELT3'] = delta / f[0].header['NAXIS3']

        # Flush updates and reopen the files
        self.close()
        self._files = self._open_fits_files(self._filenames)
        self._configure_axes()

    def real(self):
        re_gen = (np.expand_dims(re[0].data.T, axis=3)
            for re, im in self._files.itervalues())
        return np.concatenate(tuple(re_gen), axis=3) 

    def imag(self):
        im_gen = (np.expand_dims(im[0].data.T, axis=3)
            for re, im in self._files.itervalues())
        return np.concatenate(tuple(im_gen), axis=3) 

    @property
    def beam_extents(self):
        return self._beam_extents
    
    @property
    def shape(self):
        return tuple(self._axes._naxis + [4])

    def close(self):
        for re, im in self._files.itervalues():
            re.close()
            im.close()

        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etrace):
        self.close()

class TestCmpVis(unittest.TestCase):
    """
    Compares visibilities produced by different versions of Montblanc, and
    with those produced by MeqTrees. This tests the brightness and phase
    matrices associated with a single point sources
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)
        montblanc.setup_test_logging()

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
                f.write('{l:.20f} {m:.20f} {i} {q} {u} {v} {spi} {rf:.20f}\n'.format(
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
            # Beam FITS file pattern
            'pybeams_fits.filename_pattern={p}'.format(p=beam_file_pattern),
            sim_script,
            '=simulate'
            ]

        # Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
        subprocess.call(cmd_list, env=env)

        # cmd_list = [pyexe,
        #     # Meqtree Pipeline script
        #     meqpipe_actual,     
        #     #'--verbose=vb=3',             
        #     # Configuration File
        #     '-c', cfg_file,                             
        #     # Configuration section
        #     '[{section}]'.format(section=cfg_section),
        #     # Measurement Set
        #     'ms_sel.msname={ms}'.format(ms=msfile),     
        #     # Tigger sky file
        #     'tiggerlsm.filename={sm}'.format(sm=tigger_sky_file), 
        #     # Output column
        #     'ms_sel.output_column={c}'.format(c=model_data_column), 
        #     # Imaging Column
        #     'img_sel.imaging_column={c}'.format(c=model_data_column),
        #     # Beam FITS file pattern
        #     'pybeams_fits.filename_pattern={p}'.format(p=beam_file_pattern),
        #     'img_sel.output_fitsname={f}'.format(f='MEQTREES.FITS'),
        #     sim_script,
        #     '=make_dirty_image'
        #     ]

        # # Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
        # subprocess.call(cmd_list, env=env)

        # cmd_list = [pyexe,
        #     # Meqtree Pipeline script
        #     meqpipe_actual,     
        #     #'--verbose=vb=3',             
        #     # Configuration File
        #     '-c', cfg_file,                             
        #     # Configuration section
        #     '[{section}]'.format(section=cfg_section),
        #     # Measurement Set
        #     'ms_sel.msname={ms}'.format(ms=msfile),     
        #     # Tigger sky file
        #     'tiggerlsm.filename={sm}'.format(sm=tigger_sky_file), 
        #     # Output column
        #     'ms_sel.output_column={c}'.format(c=model_data_column), 
        #     # Imaging Column
        #     'img_sel.imaging_column={c}'.format(c='CORRECTED_DATA'),
        #     # Beam FITS file pattern
        #     'pybeams_fits.filename_pattern={p}'.format(p=beam_file_pattern),
        #     'img_sel.output_fitsname={f}'.format(f='MONTBLANC.FITS'),
        #     sim_script,
        #     '=make_dirty_image'
        #     ]

        # # Call the meqtrees simulation script, dumping visibilities into MODEL_DATA
        # subprocess.call(cmd_list, env=env)

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


    def get_v2_output(self, slvr_cfg, **kwargs):
        # Get visibilities from the v2 solver
        slvr_cfg = slvr_cfg.copy()
        slvr_cfg.update(**kwargs)
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
            ref_wave = montblanc.constants.C / np.full(slvr.ref_wavelength.shape,
            	_REF_FREQ)
            slvr.transfer_ref_wavelength(ref_wave.astype(slvr.ref_wavelength.dtype))

            slvr.solve()

            return slvr.X2, slvr.retrieve_model_vis().transpose(1,2,3,0)

    def get_v4_output(self, slvr_cfg, **kwargs):
        # Get visibilities from the v4 solver
        gpu_slvr_cfg = slvr_cfg.copy()
        gpu_slvr_cfg.update(**kwargs)
        gpu_slvr_cfg[Options.VERSION] = Options.VERSION_FOUR

        with FitsBeam(base_beam_file) as fb:
            beam_shape = fb.shape
            gpu_slvr_cfg[Options.E_BEAM_WIDTH] = beam_shape[0]
            gpu_slvr_cfg[Options.E_BEAM_HEIGHT] = beam_shape[1]
            gpu_slvr_cfg[Options.E_BEAM_DEPTH] = beam_shape[2]

            from montblanc.impl.rime.v4.cpu.CPUSolver import CPUSolver

            with montblanc.rime_solver(gpu_slvr_cfg) as slvr:

                cpu_slvr_cfg = slvr.config().copy()
                cpu_slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_EMPTY
                cpu_slvr = CPUSolver(cpu_slvr_cfg)

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

                fb.reconfigure_frequency_axes(slvr.retrieve_frequency())

                # Create the beam from FITS file
                ebeam = np.zeros(shape=slvr.E_beam.shape, dtype=slvr.E_beam.dtype)
                ebeam.real[:] = fb.real()
                ebeam.imag[:] = fb.imag()
                slvr.transfer_E_beam(ebeam)

                # Configure the beam extents
                ll, lm, lf, ul, um, uf = fb.beam_extents

                slvr.set_beam_ll(ll)
                slvr.set_beam_lm(lm)
                slvr.set_beam_lfreq(lf)
                slvr.set_beam_ul(ul)
                slvr.set_beam_um(um)
                slvr.set_beam_ufreq(uf)

                # Set the reference frequency
                ref_freq = np.full(slvr.ref_frequency.shape, _REF_FREQ, dtype=slvr.ref_frequency.dtype)
                slvr.transfer_ref_frequency(ref_freq)

                from montblanc.solvers import copy_solver

                copy_solver(slvr, cpu_slvr)
                cpu_slvr.compute_E_beam()

                slvr.solve()

                return slvr.X2, slvr.retrieve_model_vis()

    def get_v5_output(self, slvr_cfg, **kwargs):
        # Get visibilities from the v5 solver
        slvr_cfg = slvr_cfg.copy()
        slvr_cfg.update(**kwargs)
        slvr_cfg[Options.VERSION] = Options.VERSION_FIVE

        with FitsBeam(base_beam_file) as fb:
            beam_shape = fb.shape
            slvr_cfg[Options.E_BEAM_WIDTH] = beam_shape[0]
            slvr_cfg[Options.E_BEAM_HEIGHT] = beam_shape[1]
            slvr_cfg[Options.E_BEAM_DEPTH] = beam_shape[2]

            with montblanc.rime_solver(slvr_cfg) as slvr:
                slvr.lm[:,0] = _L
                slvr.lm[:,1] = _M

                slvr.stokes[:,:,0] = _I
                slvr.stokes[:,:,1] = _Q
                slvr.stokes[:,:,2] = _U
                slvr.stokes[:,:,3] = _V
                slvr.alpha[:] = _ALPHA

                slvr.ref_frequency[:] = _REF_FREQ

                pa_sin = np.sin(slvr.parallactic_angles)[np.newaxis,:]
                pa_cos = np.cos(slvr.parallactic_angles)[np.newaxis,:]

                l = slvr.lm[:,0,np.newaxis]*pa_cos - slvr.lm[:,1,np.newaxis]*pa_sin;
                m = slvr.lm[:,0,np.newaxis]*pa_sin + slvr.lm[:,1,np.newaxis]*pa_cos;

                fb.reconfigure_frequency_axes(slvr.frequency)

                # Configure the beam from the FITS file
                slvr.E_beam.real[:] = fb.real()
                slvr.E_beam.imag[:] = fb.imag()

                # Configure the beam extents
                ll, lm, lf, ul, um, uf = fb.beam_extents

                slvr.set_beam_ll(ll)
                slvr.set_beam_lm(lm)
                slvr.set_beam_lfreq(lf)
                slvr.set_beam_ul(ul)
                slvr.set_beam_um(um)
                slvr.set_beam_ufreq(uf)

                slvr.solve()

            import pyrap.tables as pt

            query = " ".join(["FIELD_ID=0",
                "" if slvr.is_autocorrelated() else "AND ANTENNA1 != ANTENNA2",
                "ORDERBY TIME, ANTENNA1, ANTENNA2, "
                "[SELECT SPECTRAL_WINDOW_ID FROM ::DATA_DESCRIPTION][DATA_DESC_ID]"])

            # Dump visibilities in CORRECTED_DATA
            with pt.table(msfile, ack=False, readonly=False).query(query) as ms:
                ms.putcol('CORRECTED_DATA', slvr.model_vis.reshape(-1, slvr.dim_global_size('nchan'), 4))

            return slvr.X2, slvr.model_vis.copy()


    def test_cmp_visibilities(self):
        """ Test visibilities produced by montblanc and meqtrees """
        if not os.path.exists(msfile):
            raise unittest.SkipTest("MeasurementSet '{ms}' required "
                "for this test is not present".format(ms=msfile))

        slvr_cfg = montblanc.rime_solver_cfg(
            msfile=msfile,
            sources=montblanc.sources(point=1, gaussian=0, sersic=0),
            dtype='double', version=Options.VERSION_FOUR,
            mem_budget=2*1024*1024*1024)

        # Test the v4 and v5 residuals agree
        v4_chi, v4_vis = self.get_v4_output(slvr_cfg,
            vis_output=Options.VISIBILITY_OUTPUT_RESIDUALS)
        v5_chi, v5_vis = self.get_v5_output(slvr_cfg,
            vis_output=Options.VISIBILITY_OUTPUT_RESIDUALS)

        self.assertTrue(np.allclose(v4_vis, v5_vis))
        self.assertTrue(np.allclose(v4_chi, v5_chi))

        v4_chi, v4_vis = self.get_v4_output(slvr_cfg)
        v5_chi, v5_vis = self.get_v5_output(slvr_cfg)

        # Test that v4 and v5 model visibilities agree
        self.assertTrue(np.allclose(v4_vis, v5_vis))
        self.assertTrue(np.allclose(v4_chi, v5_chi))

        # Test that meqtrees agrees with v5
        try:
            meq_vis = self.get_meq_vis(slvr_cfg[Options.MS_FILE])
            self.assertTrue(np.allclose(v5_vis, meq_vis))
        except Exception as e:
            montblanc.log.exception("Unable to run MeqTrees for "
                "purposes of comparing model visibilities. "
                "This will not be treated as a test failure")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCmpVis)
    unittest.TextTestRunner(verbosity=2).run(suite)
