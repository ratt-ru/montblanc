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
import montblanc.util as mbu

from montblanc.solvers import CUDASolver
from montblanc.config import BiroSolverConfig as Options

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

class BiroSolver(CUDASolver):
    """ BIRO Solver Implementation """

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

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E Beam cube width in l coords')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E Beam cube height in m coords')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E Beam cube height in nu coords')

        from montblanc.impl.biro.v4.config import (A, P)

        self.register_properties(P)
        self.register_arrays(A)

        self._const_data = mbu.create_rime_const_data(self, self.context)

    def const_data(self):
        return self._const_data

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

        if default_ap is None:
            default_ap = self.get_default_base_ant_pairs()

        slvr = self

        newdim = lambda d: [np.newaxis for n in range(d)]

        sed = (1 if src else 0)     # Extra source dimension
        ced = (1 if chan else 0)    # Extra channel dimension
        ned = sed + ced             # Nr of extra dimensions
        all = slice(None, None, 1)  # all slice
        idx = []                    # Index we're returning
        nsrc, ntime, nchan = slvr.dim_local_size('nsrc', 'ntime', 'nchan')

        # Create the source index, [np.newaxis,:,np.newaxis,np.newaxis] + [...]
        if src is True:
            src_slice = tuple(newdim(1) + [all] + newdim(2) + newdim(ced))
            idx.append(np.arange(nsrc)[src_slice])

        # Create the time index, [np.newaxis] + [...]  + [:,np.newaxis] + [...]
        time_slice = tuple(newdim(1) + newdim(sed) +
            [all, np.newaxis] + newdim(ced))
        idx.append(np.arange(ntime)[time_slice])

        # Create the antenna pair index, [:] + [...]  + [np.newaxis,:] + [...]
        ap_slice = tuple([all] + newdim(sed) +
            [np.newaxis, all] + newdim(ced))
        idx.append(default_ap[ap_slice])

        # Create the channel index,
        # Create the antenna pair index, [np.newaxis] + [...]  + [np.newaxis,np.newaxis] + [:]
        if chan is True:
            chan_slice = tuple(newdim(1) + newdim(sed) +
                [np.newaxis, np.newaxis] + [all])
            idx.append(np.arange(nchan)[chan_slice])

        return tuple(idx)
