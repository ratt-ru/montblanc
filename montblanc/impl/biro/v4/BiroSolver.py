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

from montblanc.impl.biro.v4.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v4.gpu.RimeGaussBSum import RimeGaussBSum
from montblanc.pipeline import Pipeline

def get_pipeline(**kwargs):
    wv = kwargs.get('weight_vector', False)
    return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=wv)])

def ary_dict(name,shape,dtype,cpu=True,gpu=True):
    return {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'registrant' : 'BiroSolver',
        'gpu' : gpu,
        'cpu' : cpu,
        'shape_member' : True,
        'dtype_member' : True
    }

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
    # Note that we don't divide by speed of light here. meqtrees code operates
    # on frequency, while we're dealing with wavelengths.
    prop_dict('gauss_scale', 'ft', fwhm2int*np.sqrt(2)*np.pi),
    prop_dict('two_pi', 'ft', 2*np.pi),
    prop_dict('ref_wave', 'ft', 0.0),
    prop_dict('sigma_sqrd', 'ft', 1.0),
    prop_dict('X2', 'ft', 0.0),
    prop_dict('beam_width', 'ft', 65),
    prop_dict('beam_clip', 'ft', 1.0881),
]

# List of arrays
A = [
    # Input Arrays
    ary_dict('uvw', (3,'ntime','na'), 'ft'),
    ary_dict('ant_pairs', (2,'ntime','nbl'), np.int32),

    ary_dict('lm', (2,'nsrc'), 'ft'),
    ary_dict('stokes', ('nsrc','ntime', 4), 'ft'),
    ary_dict('alpha', ('nsrc','ntime'), 'ft'),
    ary_dict('gauss_shape', (3, 'ngsrc'), 'ft'),
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft'),

    ary_dict('wavelength', ('nchan',), 'ft'),
    ary_dict('point_errors', (2,'ntime','na'), 'ft'),
    ary_dict('weight_vector', (4,'ntime','nbl','nchan'), 'ft'),
    ary_dict('bayes_data', (4,'ntime','nbl','nchan'), 'ct'),

    # Result arrays
    ary_dict('jones', ('nsrc','ntime','na','nchan',4), 'ct', cpu=False),
    ary_dict('vis', (4,'ntime','nbl','nchan'), 'ct', cpu=False),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', cpu=False),

    ary_dict('X2', (1, ), 'ft'),
]

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
        # Set up a default pipeline if None is supplied
        pipeline = get_pipeline(**kwargs) if pipeline is None else pipeline

        super(BiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, nssrc=nssrc, dtype=dtype,
            pipeline=pipeline, **kwargs)

        self.register_arrays(A)
        self.register_properties(P)

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
