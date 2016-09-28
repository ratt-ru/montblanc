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

from montblanc.config import (RimeSolverConfig as Options)

from montblanc.util import (
    random_float as rf,
    random_complex as rc)

def array_dict(name, shape, dtype, **kwargs):
    D = {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'temporary' : False,
    }

    D.update(kwargs)
    return D

def prop_dict(name,dtype,default):
    return {
        'name' : name,
        'dtype' : dtype,
        'default' : default,
        'setter' : True
    }

# Set up gaussian scaling parameters
# Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
# and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
fwhm2int = 1.0/np.sqrt(np.log(256))

# List of properties
P = [
    prop_dict('sigma_sqrd', 'ft', 1.0),
    prop_dict('X2', 'ft', 0.0),
]

def default_base_ant_pairs(ctx):
    """ Compute base antenna pairs """
    auto_correlations = ctx.cfg[Options.AUTO_CORRELATIONS]

    if auto_correlations == True:
        k = 0
    elif auto_correlations == False:
        k = 1
    else:
        raise ValueError("Invalid value {ac}".format(ac=auto_correlations))

    na = ctx.dim_global_size('na')
    return (i.astype(ctx.dtype) for i in np.triu_indices(na, k))

def default_antenna1(ctx):
    ant0, ant1 = default_base_ant_pairs(ctx)
    nbl_l, nbl_u = ctx.dim_extents('nbl')
    ntime = ctx.dim_local_size('ntime')
    return np.tile(ant0[nbl_l:nbl_u], ntime).reshape(ntime, nbl_u-nbl_l)

def default_antenna2(ctx):
    ant0, ant1 = default_base_ant_pairs(ctx)
    nbl_l, nbl_u = ctx.dim_extents('nbl')
    ntime = ctx.dim_local_size('ntime')
    return np.tile(ant1[nbl_l:nbl_u], ntime).reshape(ntime, nbl_u-nbl_l)

def rand_uvw(ctx):
    distance = 10
    (ntime_l, ntime_u), (na_l, na_u) = ctx.dim_extents('ntime', 'na')
    ntime, na = ntime_u - ntime_l, na_u - na_l

    # Distribute the antenna in a circle configuration
    ant_angles = 2*np.pi*np.arange(na_l, na_u)

    # Angular difference between each antenna
    ant_angle_diff = 2*np.pi/ctx.dim_global_size('na')

    # Space the time offsets for each antenna
    time_angle = (ant_angle_diff *
        np.arange(ntime_l, ntime_u)/ctx.dim_global_size('ntime'))

    # Compute offsets per antenna and timestep
    time_ant_angles = ant_angles[np.newaxis,:]+time_angle[:,np.newaxis]

    A = np.empty(ctx.shape, ctx.dtype)
    U, V, W = A[:,:,0], A[:,:,1], A[:,:,2]
    U[:] = distance*np.cos(time_ant_angles)
    V[:] = distance*np.sin(time_ant_angles)
    W[:] = np.random.random(size=(ntime,na))*0.1

    # All antenna zero coordinate are set to (0,0,0)
    A[:,0,:] = 0

    return A

def identity_on_pols(ctx):
    """
    Returns [1, 0, 0, 1] tiled up to other dimensions
    """
    assert ctx.shape[-1] == 4

    reshape_shape = tuple(1 for a in ctx.shape[:-1]) + (ctx.shape[-1], )
    tile_shape = tuple(a for a in ctx.shape[:-1]) + (1, )
    R = np.array([1,0,0,1], dtype=ctx.dtype).reshape(reshape_shape)

    return np.tile(R, tile_shape)

def default_stokes(ctx):
    """
    Returns [1, 0, 0, 0] tiled up to other dimensions
    """
    assert ctx.shape[-1] == 4

    reshape_shape = tuple(1 for a in ctx.shape[:-1]) + (ctx.shape[-1], )
    tile_shape = tuple(a for a in ctx.shape[:-1]) + (1, )
    R = np.array([1,0,0,0], dtype=ctx.dtype).reshape(reshape_shape)

    return np.tile(R, tile_shape)

def rand_stokes(ctx):
    # Should be (nsrc, ntime, 4)
    assert len(ctx.shape) == 3 and ctx.shape[2] == 4

    A = np.empty(ctx.shape, ctx.dtype)
    I, Q, U, V = A[:,:,0], A[:,:,1], A[:,:,2], A[:,:,3]
    noise = rf(I.shape, ctx.dtype)*0.1
    Q[:] = rf(Q.shape, ctx.dtype) - 0.5
    U[:] = rf(U.shape, ctx.dtype) - 0.5
    V[:] = rf(V.shape, ctx.dtype) - 0.5
    I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)

    return A

def default_gaussian_shape(ctx):
    # Should be (3, ngsrc)
    assert len(ctx.shape) == 2 and ctx.shape[0] == 3

    A = np.empty(ctx.shape, ctx.dtype)

    if A.size == 0:
        return A

    el, em, eR = A[0,:], A[1,:], A[2,:]
    el[:] = np.zeros(el.shape, ctx.dtype)
    em[:] = np.zeros(em.shape, ctx.dtype)
    eR[:] = np.ones(eR.shape, ctx.dtype)

    return A

def rand_gaussian_shape(ctx):
    # Should be (3, ngsrc)
    assert len(ctx.shape) == 2 and ctx.shape[0] == 3

    A = np.empty(ctx.shape, ctx.dtype)

    if A.size == 0:
        return A

    el, em, eR = A[0,:], A[1,:], A[2,:]
    el[:] = np.random.random(size=el.shape)
    em[:] = np.random.random(size=em.shape)
    eR[:] = np.random.random(size=eR.shape)

    return A

def default_sersic_shape(ctx):
    # Should be (3, nssrc)
    assert len(ctx.shape) == 2 and ctx.shape[0] == 3

    A = np.empty(ctx.shape, ctx.dtype)

    if A.size == 0:
        return A

    e1, e2, eS = A[0,:], A[1,:], A[2,:]
    e1[:] = 0
    e2[:] = 0
    eS[:] = 1

    return A

def test_sersic_shape(ctx):
    # Should be (3, nssrc)
    assert len(ctx.shape) == 2 and ctx.shape[0] == 3

    A = np.empty(ctx.shape, ctx.dtype)

    if A.size == 0:
        return A

    e1, e2, eS = A[0,:], A[1,:], A[2,:]
    # Random values seem to create v. large discrepancies
    # between the CPU and GPU versions. Go with
    # non-random data here, as per Marzia's original code
    e1[:] = 0
    e2[:] = 0
    eS[:] = np.pi/648000   # 1 arcsec

    return A

_freq_low = 1e9
_freq_high = 2e9
_ref_freq = 1.5e9

# List of arrays
A = [
    # UVW Coordinates
    array_dict('uvw', ('ntime', 'na', 3), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = rand_uvw),

    array_dict('antenna1', ('ntime', 'nbl'), np.int32,
        default = default_antenna1,
        test    = default_antenna1),

    array_dict('antenna2', ('ntime', 'nbl'), np.int32,
        default = default_antenna2,
        test    = default_antenna2),

    # Frequency and Reference Frequency arrays
    # TODO: This is incorrect when channel local is not the same as channel global
    array_dict('frequency', ('nchan',), 'ft',
        default = lambda c: np.linspace(_freq_low, _freq_high, c.shape[0]),
        test    = lambda c: np.linspace(_freq_low, _freq_high, c.shape[0])),

    array_dict('ref_frequency', ('nchan',), 'ft',
        default = lambda c: np.full(c.shape, _ref_freq, c.dtype),
        test    = lambda c: np.full(c.shape, _ref_freq, c.dtype)),

    # Holographic Beam

    # Pointing errors
    array_dict('point_errors', ('ntime','na','nchan',2), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: (rf(c.shape, c.dtype)-0.5)*1e-2),

    # Antenna scaling factors
    array_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default = lambda c: np.ones(c.shape, c.dtype),
        test    = lambda c: rf(c.shape, c.dtype)),

    # Parallactic angles at each timestep for each antenna
    array_dict('parallactic_angles', ('ntime', 'na'), 'ft',
        default  = lambda c: np.zeros(c.shape, c.dtype),
        test     = lambda c: rf(c.shape, c.dtype)*np.pi),

    # Extents of the beam.
    # First 3 values are lower coordinate for (l, m, frequency)
    # while the last 3 are the upper coordinates
    array_dict('beam_extents', (6,), 'ft',
        default  = lambda c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high]),
        test     = lambda c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high])),

    array_dict('beam_freq_map', ('beam_nud',), 'ft',
        default  = lambda c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True),
        test     = lambda c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True)),
    # Beam cube
    array_dict('ebeam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        default = identity_on_pols,
        test    = lambda c: rc(c.shape, c.dtype)),

    # Direction-Independent Effects
    array_dict('gterm', ('ntime', 'na', 'nchan', 4), 'ct',
        default = identity_on_pols,
        test    = lambda c: rc(c.shape, c.dtype)),

    # Point Source Definitions
    array_dict('point_lm', ('npsrc',2), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: (rf(c.shape, c.dtype)-0.5)*1e-1),
    array_dict('point_stokes', ('npsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    array_dict('point_alpha', ('npsrc','ntime'), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: rf(c.shape, c.dtype)*0.1),

    # Gaussian Source Definitions
    array_dict('gaussian_lm', ('ngsrc',2), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: (rf(c.shape, c.dtype)-0.5)*1e-1),
    array_dict('gaussian_stokes', ('ngsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    array_dict('gaussian_alpha', ('ngsrc','ntime'), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: rf(c.shape, c.dtype)*0.1),
    array_dict('gaussian_shape', (3, 'ngsrc'), 'ft',
        default = default_gaussian_shape,
        test    = rand_gaussian_shape),

    # Sersic Source Definitions
    array_dict('sersic_lm', ('nssrc',2), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: (rf(c.shape, c.dtype)-0.5)*1e-1),
    array_dict('sersic_stokes', ('nssrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    array_dict('sersic_alpha', ('nssrc','ntime'), 'ft',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: rf(c.shape, c.dtype)*0.1),
    array_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default = default_sersic_shape,
        test    = test_sersic_shape),

    # Observation Data

    # Visibility flagging array
    array_dict('flag', ('ntime', 'nbl', 'nchan', 4), np.uint8,
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: np.random.random_integers(0, 1,
            size=c.shape).astype(np.uint8)),
    # Weight array
    array_dict('weight', ('ntime','nbl','nchan',4), 'ft',
        default = lambda c: np.ones(c.shape, c.dtype),
        test    = lambda c: rf(c.shape, c.dtype)),
    # Observed Visibilities
    array_dict('observed_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: rc(c.shape, c.dtype)),

    # Model Visibilities
    array_dict('model_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda c: np.zeros(c.shape, c.dtype),
        test    = lambda c: rc(c.shape, c.dtype)),

    # Result arrays
    array_dict('bsqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct', temporary=True),
    array_dict('cplx_phase', ('nsrc','ntime','na','nchan'), 'ct', temporary=True),
    array_dict('ejones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    array_dict('ant_jones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    array_dict('sgn_brightness', ('nsrc', 'ntime'), np.int8, temporary=True),
    array_dict('source_shape', ('nsrc', 'ntime', 'nbl', 'nchan'), 'ft', temporary=True),
    array_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', temporary=True),
]
