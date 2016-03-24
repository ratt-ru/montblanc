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
from montblanc.util import random_like as rary

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

    ary_dict('flag', (4, 'ntime', 'nbl', 'nchan'), np.int32,
        default=0,
        test=lambda slvr, ary: np.random.random_integers(
            0, 1, size=ary.shape)),

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
