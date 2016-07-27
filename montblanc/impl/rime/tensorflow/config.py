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
from montblanc.util import (
    random_float as rf,
    random_complex as rc)

def ary_dict(name, shape, dtype, **kwargs):
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

    # Width of the beam cube dimension. l, m and lambda
    # Lower l and m coordinates of the beam cube
    prop_dict('beam_ll', 'ft', -0.5),
    prop_dict('beam_lm', 'ft', -0.5),
    # Upper l and m coordinates of the beam cube
    prop_dict('beam_ul', 'ft', 0.5),
    prop_dict('beam_um', 'ft', 0.5),
    prop_dict('parallactic_angle', 'ft', 0.0),
]

def rand_uvw(slvr, ary):
    distance = 10
    ntime, na = slvr.dim_local_size('ntime', 'na')
    # Distribute the antenna in a circle configuration
    ant_angles = 2*np.pi*np.arange(na)/slvr.ft(na)
    time_angle = np.arange(ntime)/slvr.ft(ntime)
    time_ant_angles = time_angle[:,np.newaxis]*ant_angles[np.newaxis,:]

    A = np.empty(ary.shape, ary.dtype)
    U, V, W = A[:,:,0], A[:,:,1], A[:,:,2]
    U[:] = distance*np.sin(time_ant_angles)
    V[:] = distance*np.sin(time_ant_angles)
    W[:] = np.random.random(size=(ntime,na))*0.1

    # All antenna zero coordinate are set to (0,0,0)
    A[:,0,:] = 0

    return A

def identity_on_pols(slvr, ary):
    """
    Returns [1, 0, 0, 1] tiled up to other dimensions
    """
    assert ary.shape[-1] == 4

    reshape_shape = tuple(1 for a in ary.shape[:-1]) + (ary.shape[-1], )
    tile_shape = tuple(a for a in ary.shape[:-1]) + (1, )
    R = np.array([1,0,0,1], dtype=ary.dtype).reshape(reshape_shape)

    return np.tile(R, tile_shape)

def default_stokes(slvr, ary):
    """
    Returns [1, 0, 0, 0] tiled up to other dimensions
    """
    assert ary.shape[-1] == 4

    reshape_shape = tuple(1 for a in ary.shape[:-1]) + (ary.shape[-1], )
    tile_shape = tuple(a for a in ary.shape[:-1]) + (1, )
    R = np.array([1,0,0,0], dtype=ary.dtype).reshape(reshape_shape)

    return np.tile(R, tile_shape)

def rand_stokes(slvr, ary):
    # Should be (nsrc, ntime, 4)
    assert len(ary.shape) == 3 and ary.shape[2] == 4

    A = np.empty(ary.shape, ary.dtype)
    I, Q, U, V = A[:,:,0], A[:,:,1], A[:,:,2], A[:,:,3]
    noise = rf(I.shape, ary.dtype)*0.1
    Q[:] = rf(Q.shape, ary.dtype) - 0.5
    U[:] = rf(U.shape, ary.dtype) - 0.5
    V[:] = rf(V.shape, ary.dtype) - 0.5
    I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)

    return A

def default_gaussian_shape(slvr, ary):
    # Should be (3, ngsrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3

    A = np.empty(ary.shape, ary.dtype)

    if A.size == 0:
        return A

    el, em, eR = A[:,0], A[:,1], A[:,2]
    el[:] = np.zeros(el.shape, ary.dtype)
    em[:] = np.zeros(em.shape, ary.dtype)
    eR[:] = np.ones(eR.shape, ary.dtype)

    return A

def rand_gaussian_shape(slvr, ary):
    # Should be (3, ngsrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3

    A = np.empty(ary.shape, ary.dtype)

    if A.size == 0:
        return A

    el, em, eR = A[0,:], A[1,:], A[2,:]
    el[:] = np.random.random(size=el.shape)
    em[:] = np.random.random(size=em.shape)
    eR[:] = np.random.random(size=eR.shape)

    return A

def default_sersic_shape(slvr, ary):
    # Should be (3, nssrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3

    A = np.empty(ary.shape, ary.dtype)
    e1, e2, eS = A[0,:], A[1,:], A[2,:]
    # Random values seem to create v. large discrepancies
    # between the CPU and GPU versions. Go with
    # non-random data here, as per Marzia's original code
    e1[:] = 0
    e2[:] = 0
    eS[:] = np.pi/648000   # 1 arcsec

    return A

# List of arrays
A = [
    # UVW Coordinates
    ary_dict('uvw', ('ntime', 'na', 3), 'ft',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = rand_uvw),

    ary_dict('antenna1', ('ntime', 'nbl'), np.int32,
        default = lambda s, a: s.default_ant_pairs()[0],
        test    = lambda s, a: s.default_ant_pairs()[0]),

    ary_dict('antenna2', ('ntime', 'nbl'), np.int32,
        default = lambda s, a: s.default_ant_pairs()[1],
        test    = lambda s, a: s.default_ant_pairs()[1]),

    # Frequency and Reference Frequency arrays
    ary_dict('frequency', ('nchan',), 'ft',
        default = lambda s, a: np.linspace(1e9, 2e9, s.dim_local_size('nchan')),
        test    = lambda s, a: np.linspace(1e9, 2e9, s.dim_local_size('nchan'))),

    ary_dict('ref_frequency', ('nchan',), 'ft',
        default = lambda s, a: np.full(fill_value=1.5e9, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: np.full(fill_value=1.5e9, shape=a.shape, dtype=a.dtype)),

    # Holographic Beam
    ary_dict('point_errors', ('ntime','na','nchan',2), 'ft',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a: (rf(a.shape, a.dtype)-0.5)*1e-2),

    ary_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default = lambda s, a: np.ones(a.shape, a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)),

    ary_dict('ebeam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, a: rc(a.shape, a.dtype)),

    # Direction-Independent Effects
    ary_dict('gterm', ('ntime', 'na', 'nchan', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, a: rc(a.shape, a.dtype)),

    # Point Source Definitions
    ary_dict('point_lm', ('npsrc',2), 'ft',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a : (rf(a.shape, a.dtype)-0.5)*1e-1),
    ary_dict('point_stokes', ('npsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    ary_dict('point_alpha', ('npsrc','ntime'), 'ft',
        default = lambda s, a: np.full(fill_value=0.0, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)*0.1),

    # Gaussian Source Definitions
    ary_dict('gaussian_lm', ('ngsrc',2), 'ft',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a: (rf(a.shape, a.dtype)-0.5)*1e-1),
    ary_dict('gaussian_stokes', ('ngsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    ary_dict('gaussian_alpha', ('ngsrc','ntime'), 'ft',
        default = lambda s, a: np.full(fill_value=0.8, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)*0.1),
    ary_dict('gaussian_shape', (3, 'ngsrc'), 'ft',
        default = default_gaussian_shape,
        test    = rand_gaussian_shape),

    # Sersic Source Definitions
    ary_dict('sersic_lm', ('nssrc',2), 'ft',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a: (rf(a.shape, a.dtype)-0.5)*1e-1),
    ary_dict('sersic_stokes', ('nssrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),
    ary_dict('sersic_alpha', ('nssrc','ntime'), 'ft',
        default = lambda s, a: np.full(fill_value=0.8, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)*0.1),
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default = default_sersic_shape,
        test    = default_sersic_shape),

    # Observation Data
    
    # Visibility flagging array
    ary_dict('flag', ('ntime', 'nbl', 'nchan', 4), np.uint8,
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a: np.random.random_integers(0, 1,
            size=a.shape).astype(np.uint8)),
    # Weight array
    ary_dict('weight', ('ntime','nbl','nchan',4), 'ft',
        default = lambda s, a: np.ones(a.shape, a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)),
    # Observed Visibilities
    ary_dict('observed_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda s, a: np.zeros(a.shape, a.dtype),
        test    = lambda s, a: rc(a.shape, a.dtype)),

    # Result arrays
    ary_dict('bsqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct', temporary=True),
    ary_dict('ant_jones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    ary_dict('model_vis', ('ntime','nbl','nchan',4), 'ct', temporary=True),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', temporary=True),
]
