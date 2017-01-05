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

def default_base_ant_pairs(self, context):
    """ Compute base antenna pairs """
    k = 0 if context.cfg[Options.AUTO_CORRELATIONS] == True else 1
    na = context.dim_global_size('na')
    return (i.astype(context.dtype) for i in np.triu_indices(na, k))

def default_antenna1(self, context):
    ant0, ant1 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    return np.tile(ant0[bl:bu], tu-tl).reshape(tu-tl, bu-bl)

def default_antenna2(self, context):
    ant0, ant1 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    return np.tile(ant1[bl:bu], tu-tl).reshape(tu-tl, bu-bl)

def rand_uvw(self, context):
    distance = 10
    (ntime_l, ntime_u), (na_l, na_u) = context.dim_extents('ntime', 'na')
    ntime, na = ntime_u - ntime_l, na_u - na_l

    # Distribute the antenna in a circle configuration
    ant_angles = 2*np.pi*np.arange(na_l, na_u)

    # Angular difference between each antenna
    ant_angle_diff = 2*np.pi/context.dim_global_size('na')

    # Space the time offsets for each antenna
    time_angle = (ant_angle_diff *
        np.arange(ntime_l, ntime_u)/context.dim_global_size('ntime'))

    # Compute offsets per antenna and timestep
    time_ant_angles = ant_angles[np.newaxis,:]+time_angle[:,np.newaxis]

    A = np.empty(context.shape, context.dtype)
    U, V, W = A[:,:,0], A[:,:,1], A[:,:,2]
    U[:] = distance*np.cos(time_ant_angles)
    V[:] = distance*np.sin(time_ant_angles)
    W[:] = np.random.random(size=(ntime,na))*0.1

    # All antenna zero coordinate are set to (0,0,0)
    A[:,0,:] = 0

    return A

def identity_on_pols(self, context):
    """
    Returns [1, 0, 0, 1] tiled up to other dimensions
    """
    A = np.empty(context.shape, context.dtype)
    A[:,:,:] = [[[1,0,0,1]]]
    return A

def default_stokes(self, context):
    """
    Returns [1, 0, 0, 0] tiled up to other dimensions
    """
    A = np.empty(context.shape, context.dtype)
    A[:,:,:] = [[[1,0,0,0]]]
    return A

def rand_stokes(self, context):
    # Should be (nsrc, ntime, 4)
    A = np.empty(context.shape, context.dtype)
    I, Q, U, V = A[:,:,0], A[:,:,1], A[:,:,2], A[:,:,3]
    noise = rf(I.shape, context.dtype)*0.1
    Q[:] = rf(Q.shape, context.dtype) - 0.5
    U[:] = rf(U.shape, context.dtype) - 0.5
    V[:] = rf(V.shape, context.dtype) - 0.5
    I[:] = np.sqrt(Q**2 + U**2 + V**2 + noise)

    return A

def default_gaussian_shape(self, context):
    # Should be (3, ngsrc)
    A = np.empty(context.shape, context.dtype)

    if A.size != 0:
        A[:,:] = [[0],[0],[1]] # el, em, eR

    return A

def rand_gaussian_shape(self, context):
    # Should be (3, ngsrc)
    A = np.empty(context.shape, context.dtype)

    if A.size == 0:
        return A

    return np.random.random(size=(context.shape)) # el, em, eR

def default_sersic_shape(self, context):
    # Should be (3, nssrc)
    A = np.empty(context.shape, context.dtype)

    if A.size != 0:
        A[:,:] = [[0],[0],[1]] # e1, e2, eS

    return A

def test_sersic_shape(self, context):
    # Should be (3, nssrc)
    assert len(context.shape) == 2 and context.shape[0] == 3

    A = np.empty(context.shape, context.dtype)

    if A.size != 0:
        # Random values seem to create v. large discrepancies
        # between the CPU and GPU versions. Go with
        # non-random data here, as per Marzia's original code
        A[:,:] = [[0],[0],[np.pi/648000]] # e1, e2, eS

    return A

_freq_low = 1e9
_freq_high = 2e9
_ref_freq = 1.5e9

RADIANS = "Radians"
HERTZ = "Hertz"
METERS = "Meters"
JANSKYS = "Janskys"
DIMENSIONLESS = "Dimensionless"

LM_DESCRIPTION = ("(l,m) coordinates for {st} sources. "
            "Offset relative to the phase centre.")
STOKES_DESCRIPTION = ("(I,Q,U,V) Stokes parameters.")
ALPHA_DESCRIPTION = ("Power term describing the distribution of a source's flux "
            "over frequency. Distribution is calculated as (nu/nu_ref)^alpha "
            "where nu is frequency.")

# List of arrays
A = [
    # UVW Coordinates
    array_dict('uvw', ('ntime', 'na', 3), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = rand_uvw,
        input   = True,
        description = "UVW antenna coordinates, normalised "
            "relative to a reference antenna.",
        units   = METERS),

    array_dict('antenna1', ('ntime', 'nbl'), np.int32,
        default = default_antenna1,
        test    = default_antenna1,
        input   = True,
        description = "Index of the first antenna "
            "of the baseline pair."),

    array_dict('antenna2', ('ntime', 'nbl'), np.int32,
        default = default_antenna2,
        test    = default_antenna2,
        input   = True,
        description = "Index of the second antenna "
            "of the baseline pair."),

    # Frequency and Reference Frequency arrays
    # TODO: This is incorrect when channel local is not the same as channel global
    array_dict('frequency', ('nchan',), 'ft',
        default = lambda s, c: np.linspace(_freq_low, _freq_high, c.shape[0]),
        test    = lambda s, c: np.linspace(_freq_low, _freq_high, c.shape[0]),
        input   = True,
        description = "Frequency. Frequencies from multiple bands "
            "are stacked on top of each other. ",
        units   = HERTZ),

    array_dict('ref_frequency', ('nchan',), 'ft',
        default = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        test    = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        input   = True,
        description = "The reference frequency associated with the "
            " channel's band.",
        units   = HERTZ),

    # Holographic Beam

    # Pointing errors
    array_dict('point_errors', ('ntime','na','nchan', 2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-2,
        input   = True,
        description = "Pointing errors for each antenna. "
            "The components express an offset in the (l,m) plane.",
        units   = RADIANS),

    # Antenna scaling factors
    array_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default = lambda s, c: np.ones(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype),
        input   = True,
        description = "Antenna scaling factors for each antenna. "
            "The components express a scale in the (l,m) plane.",
        units   = DIMENSIONLESS),

    # Parallactic angles at each timestep for each antenna
    array_dict('parallactic_angles', ('ntime', 'na'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*np.pi,
        input   = True,
        description = "Parallactic angles for each antenna.",
        units   = RADIANS),

    # Extents of the beam.
    # First 3 values are lower coordinate for (l, m, frequency)
    # while the last 3 are the upper coordinates
    array_dict('beam_extents', (6,), 'ft',
        default = lambda s, c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high]),
        test    = lambda s, c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high]),
        input   = True,
        description = "Extents of the holographic beam cube. "
            "[l_low, m_low, freq_low, l_high, m_high, freq_high].",
        units   = "[{r}, {r}, {h}, {r}, {r}, {h}]".format(r=RADIANS, h=HERTZ)),

    array_dict('beam_freq_map', ('beam_nud',), 'ft',
        default  = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True),
        test     = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True),
        input   = True,
        description = "A map describing the frequency associated with each "
            "slice of the frequency dimension of the holographic beam cube.",
        units   = HERTZ),

    # Beam cube
    array_dict('ebeam', ('beam_lw', 'beam_mh', 'beam_nud', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, c: rc(c.shape, c.dtype),
        input   = True,
        description = "Holographic beam cube providing "
            "a discretised representation of the antenna beam pattern. "
            "Used to simulate the Direction Dependent Effects (DDE) "
            " or E term of the RIME."
            "Composed of a frequency stack of (l,m) images.",
        units   = DIMENSIONLESS),

    # Direction-Independent Effects
    array_dict('gterm', ('ntime', 'na', 'nchan', 4), 'ct',
        default = identity_on_pols,
        test    = lambda s, c: rc(c.shape, c.dtype),
        input   = True,
        description = "Array providing the Direction Independent Effects (DIE) "
            "or G term of the RIME, term for each antenna.",
        units   = DIMENSIONLESS),

    # Point Source Definitions
    array_dict('point_lm', ('npsrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        description = LM_DESCRIPTION.format(st="point"),
        units   = RADIANS),
    array_dict('point_stokes', ('npsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('point_alpha', ('npsrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),

    # Gaussian Source Definitions
    array_dict('gaussian_lm', ('ngsrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        description = LM_DESCRIPTION.format(st="gaussian"),
        units   = RADIANS),
    array_dict('gaussian_stokes', ('ngsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('gaussian_alpha', ('ngsrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),
    array_dict('gaussian_shape', (3, 'ngsrc'), 'ft',
        default = default_gaussian_shape,
        test    = rand_gaussian_shape,
        description = "Parameters describing the shape of a gaussian source. "
            "(lproj, mproj, ratio) where lproj and mproj are the projections of "
            "the major and minor ellipse axes onto the l and m axes. "
            "Ratio is ratio of the minor/major axes.",
        units   = "({r}, {r}, {d})".format(r=RADIANS, d=DIMENSIONLESS)),

    # Sersic Source Definitions
    array_dict('sersic_lm', ('nssrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        description = LM_DESCRIPTION.format(st="sersic"),
        units   = "Radians"),
    array_dict('sersic_stokes', ('nssrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('sersic_alpha', ('nssrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),
    array_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default = default_sersic_shape,
        test    = test_sersic_shape,
        description = "Parameters describing the shape of a sersic source. "
            "(e1, e2, eR). Further information is required here.",
        units   = "({r}, {r}, {d})".format(r=RADIANS, d=DIMENSIONLESS)),

    # Observation Data

    # Visibility flagging array
    array_dict('flag', ('ntime', 'nbl', 'nchan', 4), np.uint8,
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: np.random.random_integers(0, 1,
            size=c.shape).astype(np.uint8),
        input   = True,
        description = "Indicates whether a visibility should be flagged when "
            "computing a Residual or Chi-Squared value.",
        unut    = DIMENSIONLESS),
    # Weight array
    array_dict('weight', ('ntime','nbl','nchan',4), 'ft',
        default = lambda s, c: np.ones(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype),
        input   = True,
        description = "Weight applied to the difference of observed and model "
            "visibilities when computing a Chi-Squared value.",
        units   = DIMENSIONLESS),
    # Observed Visibilities
    array_dict('observed_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rc(c.shape, c.dtype),
        input   = True,
        description = "Observed visibilities, used to compute residuals and "
            "Chi-Squared values."),

    # Model Visibilities
    array_dict('model_vis', ('ntime','nbl','nchan',4), 'ct',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rc(c.shape, c.dtype),
        input   = True,
        output  = True,
        description = "Model visibilities. In the context of input, these values "
            "will be added to the model visibilities computed by the RIME. "
            "This mechanism allows visibilities to be accumulated over different "
            "models, for example. However, they are zeroed by default."
            "In the context of output, these are the RIME model visibilities "),

    # Result arrays
    array_dict('bsqrt', ('nsrc', 'ntime', 'nchan', 4), 'ct', temporary=True),
    array_dict('cplx_phase', ('nsrc','ntime','na','nchan'), 'ct', temporary=True),
    array_dict('ejones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    array_dict('ant_jones', ('nsrc','ntime','na','nchan',4), 'ct', temporary=True),
    array_dict('sgn_brightness', ('nsrc', 'ntime'), np.int8, temporary=True),
    array_dict('source_shape', ('nsrc', 'ntime', 'nbl', 'nchan'), 'ft', temporary=True),
    array_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft', temporary=True),
]
