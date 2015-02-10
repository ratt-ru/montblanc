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

        slvr = self

        newdim = lambda d: [np.newaxis for n in range(d)]

        sed = (1 if src else 0)          # Extra source dimension
        ced = (1 if chan else 0)       # Extra channel dimension
        ned = sed + ced                 # Nr of extra dimensions
        all = slice(None, None, 1)   # all slice
        idx = []                                # Index we're returning

        # Create the time index, [np.newaxis,:,np.newaxis] + [...]
        time_slice = tuple([np.newaxis, all, np.newaxis] + newdim(ned))
        idx.append(np.arange(slvr.ntime)[time_slice])

        # Create the antenna pair index, [:, np.newaxis, :] + [...]
        ap_slice = tuple([all, np.newaxis, all] + newdim(ned))
        idx.append(self.get_default_base_ant_pairs()[ap_slice])

        # Create the source index, [np.newaxis,np.newaxis,np.newaxis,:] + [...]
        if src is True:
            src_slice = tuple(newdim(3) + [all] + newdim(ced))
            idx.append(np.arange(slvr.nsrc)[src_slice])

        # Create the channel index,
        # [np.newaxis,np.newaxis,np.newaxis] + [...] + [:]
        if chan is True:
            chan_slice = tuple(newdim(3 + sed) + [all])
            idx.append(np.arange(slvr.nchan)[chan_slice])

        return tuple(idx)
