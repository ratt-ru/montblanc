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

from montblanc.impl.biro.v4.gpu.RimeEBeam import RimeEBeam
from montblanc.impl.biro.v4.gpu.RimeBSqrt import RimeBSqrt
from montblanc.impl.biro.v4.gpu.RimeEKBSqrt import RimeEKBSqrt
from montblanc.impl.biro.v4.gpu.RimeSumCoherencies import RimeSumCoherencies

from montblanc.pipeline import Pipeline

def get_pipeline(slvr_cfg):
    wv = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
    return Pipeline([RimeBSqrt(),
        RimeEBeam(),
        RimeEKBSqrt(),
        RimeSumCoherencies(weight_vector=wv)])

def ary_dict(name,shape,dtype,cpu=True,gpu=True, **kwargs):
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

# List of properties
P = [
    prop_dict('gauss_scale', 'ft', fwhm2int*np.sqrt(2)*np.pi/montblanc.constants.C),
    prop_dict('two_pi_over_c', 'ft', 2*np.pi/montblanc.constants.C),
    prop_dict('ref_freq', 'ft', 0.0),
    prop_dict('sigma_sqrd', 'ft', 1.0),
    prop_dict('X2', 'ft', 0.0),

    # Width of the beam cube dimension. l, m and lambda
    prop_dict('beam_lw', 'int', 50),
    prop_dict('beam_mh', 'int', 50),
    prop_dict('beam_nud', 'int', 50),
    # Lower l and m coordinates of the beam cube
    prop_dict('beam_ll', 'ft', -0.5),
    prop_dict('beam_lm', 'ft', -0.5),
    # Upper l and m coordinates of the beam cube
    prop_dict('beam_ul', 'ft', 0.5),
    prop_dict('beam_um', 'ft', 0.5),
    prop_dict('parallactic_angle', 'ft', 0.0),
]

# List of arrays
A = [
    # Input Arrays
    ary_dict('uvw', ('ntime','na', 3), 'ft',
        default=0),
    ary_dict('ant_pairs', (2,'ntime','nbl'), np.int32,
        default=lambda slvr, ary: slvr.get_default_ant_pairs()),

    # Source Definitions
    ary_dict('lm', ('nsrc',2), 'ft',
        default=0),
    ary_dict('stokes', ('nsrc','ntime', 4), 'ft',
        default=np.array([1,0,0,0])[np.newaxis,np.newaxis,:]),
    ary_dict('alpha', ('nsrc','ntime'), 'ft',
        default=0.8),
    ary_dict('gauss_shape', (3, 'ngsrc'), 'ft',
        default=np.array([1,2,3])[:,np.newaxis]),
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default=np.array([1,1,1],np.int32)[:,np.newaxis]),

    ary_dict('frequency', ('nchan',), 'ft',
        default=lambda slvr, ary: np.linspace(1e9, 2e9, slvr.nchan)),

    # Beam
    ary_dict('point_errors', ('ntime','na','nchan',2), 'ft',
        default=0),
    ary_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default=1),
    ary_dict('E_beam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        default=np.array([1,0,0,1])[np.newaxis,np.newaxis,np.newaxis,:]),

    # Direction-Independent Effects
    ary_dict('G_term', ('ntime', 'na', 'nchan', 4), 'ct',
        default=np.array([1,0,0,1])[np.newaxis,np.newaxis,np.newaxis,:]),

    ary_dict('weight_vector', ('ntime','nbl','nchan',4), 'ft',
        default=1),
    ary_dict('bayes_data', ('ntime','nbl','nchan',4), 'ct',
        default=0),

    # Result arrays
    ary_dict('B_sqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct', cpu=False),
    ary_dict('jones', ('nsrc','ntime','na','nchan',4), 'ct', cpu=False),
    ary_dict('vis', ('ntime','nbl','nchan',4), 'ct', cpu=False),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', cpu=False),

    ary_dict('X2', (1, ), 'ft'),
]

class BiroSolver(BaseSolver):
    """ Shared Data implementation for BIRO """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
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
        return np.int32(np.triu_indices(self.na, 1))

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, ntime, nbl), dtype=np.int32])
        containing the default antenna pairs for each timestep
        at each baseline.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna.
        return np.tile(self.get_default_base_ant_pairs(), self.ntime) \
            .reshape(2, self.ntime, self.nbl)

    def get_ap_idx(self, src=False, chan=False):
        """
        This method produces an index
        which arranges per antenna values into a
        per baseline configuration, using the default
        per timestep and baseline antenna pair configuration.
        Thus, indexing an array with shape (na) will produce
        a view of the values in this array with shape (2, nbl).

        Consequently, this method is suitable for indexing
        an array of shape (ntime, na). Specifiying source
        and channel dimensions allows indexing of an array
        of shape (nsrc, ntime, na, nchan).

        Using this index on an array of (ntime, na)
        produces a (2, ntime, nbl) array,
        or (2, nsrc, ntime, nbl, nchan) if source
        and channel are also included.

        The values for the first antenna are in position 0, while
        those for the second are in position 1.

        >>> ap = slvr.get_ap_idx()
        >>> u_ant = np.random.random(size=(ntime,na))
        >>> u_bl = u_ant[ap][1] - u_ant[ap][0]
        >>> assert u_bl.shape == (2, ntime, nbl)
        """

        slvr = self

        newdim = lambda d: [np.newaxis for n in range(d)]

        sed = (1 if src else 0)          # Extra source dimension
        ced = (1 if chan else 0)       # Extra channel dimension
        ned = sed + ced                 # Nr of extra dimensions
        all = slice(None, None, 1)   # all slice
        idx = []                                # Index we're returning

        # Create the source index, [np.newaxis,:,np.newaxis,np.newaxis] + [...]
        if src is True:
            src_slice = tuple(newdim(1) + [all] + newdim(2) + newdim(ced))
            idx.append(np.arange(slvr.nsrc)[src_slice])

        # Create the time index, [np.newaxis] + [...]  + [:,np.newaxis] + [...]
        time_slice = tuple(newdim(1) + newdim(sed) +
            [all, np.newaxis] + newdim(ced))
        idx.append(np.arange(slvr.ntime)[time_slice])

        # Create the antenna pair index, [:] + [...]  + [np.newaxis,:] + [...]
        ap_slice = tuple([all] + newdim(sed) +
            [np.newaxis, all] + newdim(ced))
        idx.append(self.get_default_base_ant_pairs()[ap_slice])

        # Create the channel index,
        # Create the antenna pair index, [np.newaxis] + [...]  + [np.newaxis,np.newaxis] + [:]
        if chan is True:
            chan_slice = tuple(newdim(1) + newdim(sed) +
                [np.newaxis, np.newaxis] + [all])
            idx.append(np.arange(slvr.nchan)[chan_slice])

        return tuple(idx)
