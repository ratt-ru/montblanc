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

from montblanc.BaseSolver import BaseSolver
from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_NSSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

class BiroSolver(BaseSolver):
    """ Shared Data implementation for BIRO """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, nssrc=DEFAULT_NSSRC,
        dtype=DEFAULT_DTYPE, pipeline=None, **kwargs):
        """
        BiroSolver Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            nssrc : integer
                Number of sersic (exponential) sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
            pipeline : list of nodes
                nodes defining the GPU kernels used to solve this RIME
        Keyword Arguments:
            context : pycuda.context.Context
                CUDA context to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSolver object.
        """

        # Turn off auto_correlations
        kwargs['auto_correlations'] = False

        super(BiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, nssrc=nssrc, dtype=dtype, pipeline=pipeline, **kwargs)

        slvr = self
        na, nbl, nchan, ntime = slvr.na, slvr.nbl, slvr.nchan, slvr.ntime
        npsrc, ngsrc, nssrc, nsrc = slvr.npsrc, slvr.ngsrc, slvr.nssrc, slvr.nsrc
        ft, ct = slvr.ft, slvr.ct

        # Curry the register_array function for simplicity
        def reg(name,shape,dtype):
            self.register_array(name=name,shape=shape,dtype=dtype,
                registrant='BaseSolver', gpu=True, cpu=False,
                shape_member=True, dtype_member=True)

        def reg_prop(name,dtype,default):
            self.register_property(name=name,dtype=dtype,
                default=default,registrant='BaseSolver', setter=True)

        # Set up gaussian scaling parameters
        # Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
        # and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
        fwhm2int = 1.0/np.sqrt(np.log(256))
        # Note that we don't divide by speed of light here. meqtrees code operates
        # on frequency, while we're dealing with wavelengths.
        reg_prop('gauss_scale', ft, fwhm2int*np.sqrt(2)*np.pi)
        reg_prop('ref_wave', ft, 0.0)
        reg_prop('two_pi', ft, 2*np.pi)

        reg_prop('sigma_sqrd', ft, 1.0)
        reg_prop('X2', ft, 0.0)
        reg_prop('beam_width', ft, 65)
        reg_prop('beam_clip', ft, 1.0881)

        reg(name='uvw', shape=(3,'ntime','na'), dtype=ft)
        reg(name='ant_pairs', shape=(2,'ntime','nbl'), dtype=np.int32)

        reg(name='lm', shape=(2,'nsrc'), dtype=ft)
        reg(name='brightness', shape=(5,'ntime','nsrc'), dtype=ft)
        reg(name='gauss_shape', shape=(3, 'ngsrc'), dtype=ft)
        reg(name='sersic_shape', shape=(3, 'nssrc'), dtype=ft)

        reg(name='wavelength', shape=('nchan',), dtype=ft)
        reg(name='point_errors', shape=(2,'ntime','na'), dtype=ft)
        reg(name='weight_vector', shape=(4,'ntime','nbl','nchan'), dtype=ft)
        reg(name='bayes_data', shape=(4,'ntime','nbl','nchan'), dtype=ct)

        reg(name='jones_scalar', shape=('ntime','na','nsrc','nchan'), dtype=ct)
        reg(name='vis', shape=(4,'ntime','nbl','nchan'), dtype=ct)
        reg(name='chi_sqrd_result', shape=('ntime','nbl','nchan'), dtype=ft)

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, ntime, nbl), dtype=np.int32]) containing the
        default antenna pairs for each timestep at each baseline.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna.
        slvr = self

        return np.tile(np.int32(np.triu_indices(slvr.na,1)),
            slvr.ntime).reshape(2,slvr.ntime,slvr.nbl)

    def get_flat_ap_idx(self, src=False, chan=False):
        """
        Returns a flattened antenna pair index

        Parameters
        ----------
        src : boolean
            Expand the index over the source dimension
        chan : boolean
            Expand the index over the channel dimension
        """
        # TODO: Test for src=False and chan=True, and src=True and chan=False
        # This works for
        # - src=True and chan=True.
        # - src=False and chan=False.

        # The flattened antenna pair array will look something like this.
        # It is based on 2 x ntime x nbl. Here we have 3 baselines and
        # 4 timesteps.
        #
        #            timestep
        #       0 0 0 1 1 1 2 2 2 3 3 3
        #
        # ant1: 0 0 1 0 0 1 0 0 1 0 0 1
        # ant2: 1 2 2 1 2 2 1 2 2 1 2 2

        slvr = self
        ap = slvr.get_default_ant_pairs().reshape(2,slvr.ntime*slvr.nbl)

        C = 1

        if src is True: C *= slvr.nsrc
        if chan is True: C *= slvr.nchan

        repeat = np.repeat(np.arange(slvr.ntime),slvr.nbl)*slvr.na*C

        ant0 = ap[0]*C + repeat
        ant1 = ap[1]*C + repeat

        if src is True or chan is True:
            tile = np.tile(np.arange(C),slvr.ntime*slvr.nbl)

            ant0 = np.repeat(ant0, C) + tile
            ant1 = np.repeat(ant1, C) + tile

        return ant0, ant1
