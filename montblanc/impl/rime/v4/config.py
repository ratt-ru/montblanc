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

def ary_dict(name, shape, dtype, **kwargs):
    D = {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'registrant' : 'BiroSolver',
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
    prop_dict('ref_freq', 'ft', 1.5e9),
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

    ary[:,:,0] = distance*np.sin(time_ant_angles)                # U
    ary[:,:,1] = distance*np.sin(time_ant_angles)                # V
    ary[:,:,2] = np.random.random(size=(ntime,na))*0.1 # W

    # All antenna zero coordinate are set to (0,0,0)
    ary[:,0,:] = 0

    return ary

def rand_stokes(slvr, ary):
    I, Q, U, V = ary[:,:,0], ary[:,:,1], ary[:,:,2], ary[:,:,3]
    noise = np.random.random(size=I.shape)*0.1
    Q[:] = np.random.random(size=Q.shape) - 0.5
    U[:] = np.random.random(size=U.shape) - 0.5
    V[:] = np.random.random(size=V.shape) - 0.5
    I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)

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

from enum import Enum

# String constants for classifying arrays
class Classifier(Enum):
    X2_INPUT = 'X2_input'
    GPU_SCRATCH = 'gpu_scratch'
    SIMULATOR_OUTPUT = 'simulator_output'

    B_SQRT_INPUT = 'b_sqrt_input'
    E_BEAM_INPUT = 'e_beam_input'
    EKB_SQRT_INPUT = 'ekb_sqrt_input'
    COHERENCIES_INPUT = 'coherencies_input'

# List of arrays
A = [
    # Input Arrays
    ary_dict('uvw', ('ntime','na', 3), 'ft',
        classifiers=frozenset([Classifier.EKB_SQRT_INPUT,
            Classifier.COHERENCIES_INPUT]),
        default=0,
        test=rand_uvw),

    ary_dict('ant_pairs', (2,'ntime','nbl'), np.int32,
        classifiers=frozenset([Classifier.COHERENCIES_INPUT]),
        default=lambda slvr, ary: slvr.default_ant_pairs(),
        test=lambda slvr, ary: slvr.default_ant_pairs()),

    ary_dict('frequency', ('nchan',), 'ft',
        classifiers=frozenset([Classifier.B_SQRT_INPUT,
            Classifier.EKB_SQRT_INPUT,
            Classifier.COHERENCIES_INPUT]),
        default=lambda slvr, ary: np.linspace(1e9, 2e9, slvr.dim_local_size('nchan')),
        test=lambda slvr, ary: np.linspace(1e9, 2e9, slvr.dim_local_size('nchan'))),

    # Holographic Beam
    ary_dict('point_errors', ('ntime','na','nchan',2), 'ft',
        classifiers=frozenset([Classifier.E_BEAM_INPUT]),
        default=0,
        test=lambda slvr, ary: (rary(ary) - 0.5)*1e-2),

    ary_dict('antenna_scaling', ('na','nchan',2), 'ft',
        classifiers=frozenset([Classifier.E_BEAM_INPUT]),
        default=1,
        test=lambda slvr, ary: rary(ary)),

    ary_dict('E_beam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        classifiers=frozenset([Classifier.E_BEAM_INPUT]),
        default=np.array([1,0,0,1])[np.newaxis,np.newaxis,np.newaxis,:],
        test=lambda slvr, ary: rary(ary)),

    # Direction-Independent Effects
    ary_dict('G_term', ('ntime', 'na', 'nchan', 4), 'ct',
        classifiers=frozenset([Classifier.COHERENCIES_INPUT]),
        default=np.array([1,0,0,1])[np.newaxis,np.newaxis,np.newaxis,:],
        test=lambda slvr, ary: rary(ary)),

    # Source Definitions
    ary_dict('lm', ('nsrc',2), 'ft',
        classifiers=frozenset([Classifier.E_BEAM_INPUT,
            Classifier.EKB_SQRT_INPUT]),
        default=0,
        test=lambda slvr, ary: (rary(ary)-0.5) * 1e-1),

    ary_dict('stokes', ('nsrc','ntime', 4), 'ft',
        classifiers=frozenset([Classifier.B_SQRT_INPUT]),
        default=np.array([1,0,0,0])[np.newaxis,np.newaxis,:],
        test=rand_stokes),

    ary_dict('alpha', ('nsrc','ntime'), 'ft',
        classifiers=frozenset([Classifier.B_SQRT_INPUT]),
        default=0.8,
        test=lambda slvr, ary: rary(ary)*0.1),

    ary_dict('gauss_shape', (3, 'ngsrc'), 'ft',
        classifiers=frozenset([Classifier.COHERENCIES_INPUT]),
        default=np.array([0,0,1])[:,np.newaxis],
        test=rand_gauss_shape),
    
    ary_dict('sersic_shape', (3, 'nssrc'), 'ft',
        classifiers=frozenset([Classifier.COHERENCIES_INPUT]),
        default=np.array([0,0,0])[:,np.newaxis],
        test=rand_sersic_shape),

    # Visibility flagging arrays
    ary_dict('flag', ('ntime', 'nbl', 'nchan', 4), np.uint8,
        classifiers=frozenset([Classifier.X2_INPUT,
            Classifier.COHERENCIES_INPUT]),
        default=0,
        test=lambda slvr, ary: np.random.random_integers(
            0, 1, size=ary.shape)),

    # Bayesian Data
    ary_dict('weight_vector', ('ntime','nbl','nchan',4), 'ft',
        classifiers=frozenset([Classifier.X2_INPUT,
            Classifier.COHERENCIES_INPUT]),
        default=1,
        test=lambda slvr, ary: rary(ary)),
    ary_dict('observed_vis', ('ntime','nbl','nchan',4), 'ct',
        classifiers=frozenset([Classifier.X2_INPUT,
            Classifier.COHERENCIES_INPUT]),
        default=0,
        test=lambda slvr, ary: rary(ary)),

    # Result arrays
    ary_dict('B_sqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct',
        classifiers=frozenset([Classifier.GPU_SCRATCH])),
    ary_dict('jones', ('nsrc','ntime','na','nchan',4), 'ct',
        classifiers=frozenset([Classifier.GPU_SCRATCH])),
    ary_dict('model_vis', ('ntime','nbl','nchan',4), 'ct',
        classifiers=frozenset([Classifier.SIMULATOR_OUTPUT])),
    ary_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft',
        classifiers=frozenset([Classifier.GPU_SCRATCH])),
]
