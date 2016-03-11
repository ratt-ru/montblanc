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

import numpy as np

import montblanc

from montblanc.BaseSolver import BaseSolver
from montblanc.config import BiroSolverConfigurationOptions as Options

from montblanc.impl.biro.v2.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v2.gpu.RimeGaussBSum import RimeGaussBSum
from montblanc.pipeline import Pipeline
from montblanc.util import random_like as rary

def get_pipeline(slvr_cfg):
    wv = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
    return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=wv)])

def ary_dict(name, shape, dtype, cpu=False, gpu=True, **kwargs):
    D = {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'registrant' : 'BiroSolver',
        'gpu' : gpu,
        'cpu' : cpu,
        'shape_member' : True,
        'dtype_member' : True
    }

    D.update(kwargs)
    return D

def prop_dict(name,dtype,default):
    return {
        'name' : name,
        'dtype' : dtype,
        'default' : default,
        'registrant' : 'BiroSolver',
        'setter' : True
    }

# Set up gaussian scaling parameters
# Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
# and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
fwhm2int = 1.0/np.sqrt(np.log(256))

# Dictionary of properties
P = [
    # Note that we don't divide by speed of light here. meqtrees code operates
    # on frequency, while we're dealing with wavelengths.
    prop_dict('gauss_scale', 'ft', fwhm2int*np.sqrt(2)*np.pi),
    prop_dict('two_pi', 'ft', 2*np.pi),
    prop_dict('ref_wave', 'ft', 1.5e9),
    prop_dict('sigma_sqrd', 'ft', 1.0),
    prop_dict('X2', 'ft', 0.0),
    prop_dict('beam_width', 'ft', 65),
    prop_dict('beam_clip', 'ft', 1.0881),
]

def rand_uvw(slvr, ary):
    ntime, na = slvr.dim_local_size('ntime', 'na')
    distance = 10
    # Distribute the antenna in a circle configuration
    ant_angles = 2*np.pi*np.arange(na)/slvr.ft(na)
    time_angle = np.arange(ntime)/slvr.ft(ntime)
    time_ant_angles = time_angle[:,np.newaxis]*ant_angles[np.newaxis,:]

    ary[0,:,:] = distance*np.sin(time_ant_angles)                # U
    ary[1,:,:] = distance*np.sin(time_ant_angles)                # V
    ary[2,:,:] = np.random.random(size=(ntime, na))*0.1 # W

    # All antenna zero coordinate are set to (0,0,0)
    ary[:,:0] = 0

    return ary

def rand_gauss_shape(slvr, ary):
    el, em, eR = ary[0,:], ary[1,:], ary[2,:]
    el[:] = np.random.random(size=el.shape)
    em[:] = np.random.random(size=em.shape)
    eR[:] = np.random.random(size=eR.shape)

    return ary

def rand_sersic_shape(slvr, ary):
    e1, e2, eS = ary[0,:], ary[1,:], ary[2,:]
    # Random values seem to create v. large discrepancies
    # between the CPU and GPU versions. Go with
    # non-random data here, as per Marzia's original code
    e1[:] = 0
    e2[:] = 0
    eS[:] = np.pi/648000   # 1 arcsec

    return ary

# Dictionary of arrays
A = [
    # Input Arrays
    ary_dict('uvw', (3,'ntime','na'), 'ft',
        default=0,
        test=rand_uvw),

    ary_dict('ant_pairs', (2,'ntime','nbl'), np.int32,
        default=lambda slvr, ary: slvr.get_default_ant_pairs(),
        test=lambda slvr, ary: slvr.get_default_ant_pairs()),

    ary_dict('lm', (2,'nsrc'), 'ft',
        default=0,
        test=lambda slvr, ary: (rary(ary)-0.5)*1e-1),

    ary_dict('brightness', (5,'ntime','nsrc'), 'ft',
        default=np.array([1,0,0,1,0.8])[:,np.newaxis,np.newaxis],
        test=lambda slvr, ary: (rary(ary))),

    ary_dict('gauss_shape', (3, 'ngsrc'), 'ft',
        default=np.array([0,0,1])[:,np.newaxis],
        test=rand_gauss_shape),
    
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default=np.array([0,0,0])[:,np.newaxis],
        test=rand_sersic_shape),

    ary_dict('wavelength', ('nchan',), 'ft',
        default=lambda slvr, ary: montblanc.constants.C / \
            np.linspace(1e9, 2e9, slvr.dim_local_size('nchan')),
        test=lambda slvr, ary: montblanc.constants.C / \
            np.linspace(1e9, 2e9, slvr.dim_local_size('nchan'))),

    ary_dict('point_errors', (2,'ntime','na'), 'ft',
        default=1,
        test=lambda slvr, ary: (rary(ary)-0.5)*1e-2),

    ary_dict('weight_vector', (4,'ntime','nbl','nchan'), 'ft',
        default=1,
        test=lambda slvr, ary: rary(ary)),
    ary_dict('bayes_data', (4,'ntime','nbl','nchan'), 'ct',
        default=0,
        test=lambda slvr, ary: rary(ary)),

    # Result arrays
    ary_dict('jones_scalar', ('ntime','na','nsrc','nchan'), 'ct'),
    ary_dict('vis', (4,'ntime','nbl','nchan'), 'ct'),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft'),

    ary_dict('X2', (1, ), 'ft'),
]

class BiroSolver(BaseSolver):
    """ Solver implementation for BIRO """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : BiroSolverConfiguration
        """

        # Set up a default pipeline if None is supplied
        slvr_cfg.setdefault('pipeline', get_pipeline(slvr_cfg))

        super(BiroSolver, self).__init__(slvr_cfg)

        self.register_properties(P)
        self.register_arrays(A)

    def get_default_base_ant_pairs(self):
        """
        Return an np.array(shape=(2, nbl), dtype=np.int32]) containing the
        default antenna pairs for each baseline.
        """
        na = self.dim_local_size('na')
        return np.int32(np.triu_indices(na, 1))

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, ntime, nbl), dtype=np.int32])
        containing the default antenna pairs for each timestep
        at each baseline.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna.
        ntime, nbl = self.dim_local_size('ntime', 'nbl')
        return np.tile(self.get_default_base_ant_pairs(), ntime) \
            .reshape(2, ntime, nbl)

    def get_ap_idx(self, default_ap=None, src=False, chan=False):
        """
        This method produces an index
        which arranges per antenna values into a
        per baseline configuration, using the supplied (default_ap)
        per timestep and baseline antenna pair configuration.
        Thus, indexing an array with shape (na) will produce
        a view of the values in this array with shape (2, nbl).

        Consequently, this method is suitable for indexing
        an array of shape (ntime, na). Specifiying source
        and channel dimensions allows indexing of an array
        of shape (ntime, na, nsrc, nchan).

        Using this index on an array of (ntime, na)
        produces a (2, ntime, nbl) array,
        or (2, ntime, nbl, nsrc, nchan) if source
        and channel are also included.

        The values for the first antenna are in position 0, while
        those for the second are in position 1.

        >>> ap = slvr.get_ap_idx()
        >>> u_ant = np.random.random(size=(ntime,na))
        >>> u_bl = u_ant[ap][1] - u_ant[ap][0]
        >>> assert u_bl.shape == (2, ntime, nbl)
        """

        if default_ap is None:
            default_ap = self.get_default_base_ant_pairs()

        newdim = lambda d: [np.newaxis for n in range(d)]
        nsrc, ntime, nchan = self.dim_local_size('nsrc', 'ntime', 'nchan')

        sed = (1 if src else 0)      # Extra source dimension
        ced = (1 if chan else 0)     # Extra channel dimension
        ned = sed + ced              # Nr of extra dimensions
        all = slice(None, None, 1)   # all slice
        idx = []                     # Index we're returning

        # Create the time index, [np.newaxis,:,np.newaxis] + [...]
        time_slice = tuple([np.newaxis, all, np.newaxis] + newdim(ned))
        idx.append(np.arange(ntime)[time_slice])

        # Create the antenna pair index, [:, np.newaxis, :] + [...]
        ap_slice = tuple([all, np.newaxis, all] + newdim(ned))
        idx.append(default_ap[ap_slice])

        # Create the source index, [np.newaxis,np.newaxis,np.newaxis,:] + [...]
        if src is True:
            src_slice = tuple(newdim(3) + [all] + newdim(ced))
            idx.append(np.arange(nsrc)[src_slice])

        # Create the channel index,
        # [np.newaxis,np.newaxis,np.newaxis] + [...] + [:]
        if chan is True:
            chan_slice = tuple(newdim(3 + sed) + [all])
            idx.append(np.arange(nchan)[chan_slice])

        return tuple(idx)
