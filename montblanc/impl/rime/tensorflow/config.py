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
import tensorflow as tf

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
    # Distribute the antenna in a circular configuration
    ant_angles = 2.*np.pi*tf.cast(tf.range(na), dtype=ary.dtype)/na
    time_angles = tf.cast(tf.range(ntime), dtype=ary.dtype)/ntime
    time_ant_angles = (
        tf.reshape(time_angles, [ntime,1,1])*
        tf.reshape(ant_angles, [1,na,1]))

    U = distance*tf.sin(time_ant_angles)
    V = distance*tf.cos(time_ant_angles)
    W = rf((ntime,na,1), U.dtype)

    # All antenna zero coordinate are set to (0,0,0)
    # These assigns need to part of the graph
    # Rather use a tf.select here...
    #Uassign = tf.slice(U, [0,0,0], [-1, 1, -1]).assign(0)
    #Vassign = tf.slice(V, [0,0,0], [-1, 1, -1]).assign(0)
    #Wassign = tf.slice(W, [0,0,0], [-1, 1, -1]).assign(0)

    return tf.concat(2, [U, V, W])

def identity_on_pols(slvr, ary):
    """
    Returns [1, 0, 0, 1] tiled up to other dimensions
    """
    assert ary.shape[-1] == 4

    reshape_shape = tuple(1 for a in ary.shape[:-1]) + (ary.shape[-1], )
    tile_shape = tuple(a for a in ary.shape[:-1]) + (1, )
    R = tf.reshape([1,0,0,1], reshape_shape)

    return tf.tile(R, tile_shape)

def default_stokes(slvr, ary):
    """
    Returns [1, 0, 0, 0] tiled up to other dimensions
    """
    assert ary.shape[-1] == 4

    reshape_shape = tuple(1 for a in ary.shape[:-1]) + (ary.shape[-1], )
    tile_shape = tuple(a for a in ary.shape[:-1]) + (1, )
    R = tf.reshape([1,0,0,0], reshape_shape)

    return tf.tile(R, tile_shape)

def rand_stokes(slvr, ary):
    # Should be (nsrc, ntime, 4)
    assert len(ary.shape) == 3 and ary.shape[2] == 4
    # Want (nsrc, ntime, 1) for the concatentation
    shape = ary.shape[:-1] + (1,)

    Q = rf(shape, ary.dtype) - 0.5
    U = rf(shape, ary.dtype) - 0.5
    V = rf(shape, ary.dtype) - 0.5
    noise = rf(shape, ary.dtype)*0.1
    I = tf.sqrt(Q**2 + U**2 + V**2 + noise)

    return tf.concat(2, [I, Q, U, V])

def default_gauss_shape(slvr, ary):
    # Should be (3, ngsrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3
    # Want (1, ngsrc) for the concatenation
    shape = (1,) + ary.shape[1:]

    el = tf.zeros(shape, ary.dtype)
    em = tf.zeros(shape, ary.dtype)
    eR = tf.ones(shape, ary.dtype)

    return tf.concat(0, [el, em, eR])

def rand_gauss_shape(slvr, ary):
    # Should be (3, ngsrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3
    # Want (1, ngsrc) for the concatenation
    shape = (1,) + ary.shape[1:]
    el = rf(shape, ary.dtype)
    em = rf(shape, ary.dtype)
    eR = rf(shape, ary.dtype)

    return tf.concat(0, [el, em, eR])

def rand_sersic_shape(slvr, ary):
    # Should be (3, nssrc)
    assert len(ary.shape) == 2 and ary.shape[0] == 3
    # Want (1, nssrc) for the concatenation
    shape = (1,) + ary.shape[1:]
    # Random values seem to create v. large discrepancies
    # between the CPU and GPU versions. Go with
    # non-random data here, as per Marzia's original code

    e1 = tf.zeros(shape, ary.dtype)
    e2 = tf.zeros(shape, ary.dtype)
    eS = tf.constant(np.pi/648000, shape=shape, dtype=ary.dtype)

    return tf.concat(0, [e1, e2, eS])

# List of arrays
A = [
    # Input Arrays
    ary_dict('uvw', ('ntime', 'na', 3), 'ft',
        default = lambda s, a: tf.zeros(a.shape, a.dtype),
        test    = rand_uvw),

    ary_dict('antenna1', ('ntime', 'nbl'), np.int32,
        default = lambda s, a: s.default_ant_pairs()[0],
        test    = lambda s, a: s.default_ant_pairs()[0]),

    ary_dict('antenna2', ('ntime', 'nbl'), np.int32,
        default = lambda s, a: s.default_ant_pairs()[1],
        test    = lambda s, a: s.default_ant_pairs()[1]),

    ary_dict('frequency', ('nchan',), 'ft',
        default = lambda s, a: tf.linspace(1e9, 2e9, s.dim_local_size('nchan')),
        test    = lambda s, a: tf.linspace(1e9, 2e9, s.dim_local_size('nchan'))),

    ary_dict('ref_frequency', ('nchan',), 'ft',
        default = lambda s, a: tf.constant(1.5e9, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: tf.constant(1.5e9, shape=a.shape, dtype=a.dtype)),

    # Holographic Beam
    ary_dict('point_errors', ('ntime','na','nchan',2), 'ft',
        default = lambda s, a: tf.zeros(a.shape, a.dtype),
        test    = lambda s, a: (rf(a.shape, a.dtype)-0.5)*1e-2),

    ary_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default = lambda s, a: tf.ones(a.shape, a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)),

    ary_dict('ebeam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, a : rc(a.shape, a.dtype)),

    # Direction-Independent Effects
    ary_dict('gterm', ('ntime', 'na', 'nchan', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, a: rc(a.shape, a.dtype)),

    # Source Definitions
    ary_dict('lm', ('nsrc',2), 'ft',
        default = lambda s, a: tf.zeros(a.shape, a.dtype),
        test    = lambda s, a : (rf(a.shape, a.dtype)-0.5)*1e-1),

    ary_dict('stokes', ('nsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes),

    ary_dict('alpha', ('nsrc','ntime'), 'ft',
        default = lambda s, a: tf.constant(0.8, shape=a.shape, dtype=a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)*0.1),

    ary_dict('gauss_shape', (3, 'ngsrc'), 'ft',
        default = default_gauss_shape,
        test    = rand_gauss_shape),
    
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default = rand_sersic_shape,
        test    = rand_sersic_shape),

    # Visibility flagging arrays
    ary_dict('flag', ('ntime', 'nbl', 'nchan', 4), np.uint8,
        default = lambda s, a: tf.zeros(a.shape, a.dtype),
        test    = lambda s, a: tf.random_uniform(shape=a.shape,
            minval=0, maxval=2, dtype=np.int32)),

    # Observed Visibility Data
    ary_dict('weight', ('ntime','nbl','nchan',4), 'ft',
        default = lambda s, a: tf.ones(a.shape, a.dtype),
        test    = lambda s, a: rf(a.shape, a.dtype)),
    ary_dict('observed_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda s, a: tf.zeros(a.shape, a.dtype),
        test    = lambda s, a: rc(a.shape, a.dtype)),

    # Result arrays
    ary_dict('bsqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct', temporary=True),
    ary_dict('ant_jones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    ary_dict('model_vis', ('ntime','nbl','nchan',4), 'ct', temporary=True),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', temporary=True),
]
