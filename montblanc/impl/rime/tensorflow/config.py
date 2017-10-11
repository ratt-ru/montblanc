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

from montblanc.util import (
    random_float as rf,
    random_complex as rc)

def array_dict(name, shape, dtype, **kwargs):
    tags = kwargs.pop('tags', ())

    if isinstance(tags, str):
        tags = set(s.strip() for s in tags.split(","))
    else:
        tags = set(str(s).strip() for s in tags)

    D = {
        'name' : name,
        'shape' : shape,
        'dtype' : dtype,
        'tags' : tags,
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
    k = 0 if context.cfg['auto_correlations'] == True else 1
    na = context.dim_global_size('na')
    gen = (i.astype(context.dtype) for i in np.triu_indices(na, k))

    # Cache np.triu_indices(na, k) as its likely that (na, k) will
    # stay constant much of the time. Assumption here is that this
    # method will be grafted onto a DefaultsSourceProvider with
    # the appropriate members.
    if self._is_cached:
        array_cache = self._chunk_cache['default_base_ant_pairs']
        key = (k, na)

        # Cache miss
        if key not in array_cache:
            array_cache[key] = tuple(gen)

        return array_cache[key]

    return tuple(gen)

def default_antenna1(self, context):
    """ Default antenna1 values """
    ant1, ant2 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    ant1_result = np.empty(context.shape, context.dtype)
    ant1_result[:,:] = ant1[np.newaxis,bl:bu]
    return ant1_result

def default_antenna2(self, context):
    """ Default antenna2 values """
    ant1, ant2 = default_base_ant_pairs(self, context)
    (tl, tu), (bl, bu) = context.dim_extents('ntime', 'nbl')
    ant2_result = np.empty(context.shape, context.dtype)
    ant2_result[:,:] = ant2[np.newaxis,bl:bu]
    return ant2_result

def rand_uvw(self, context):
    distance = 10
    (tl, tu), (al, au) = context.dim_extents('ntime', 'na')
    ntime, na = tu - tl, au - al

    # Distribute the antenna in a circle configuration
    ant_angles = 2*np.pi*np.arange(al, au)

    # Angular difference between each antenna
    ant_angle_diff = 2*np.pi/context.dim_global_size('na')

    # Space the time offsets for each antenna
    time_angle = (ant_angle_diff *
        np.arange(tl, tu)/context.dim_global_size('ntime'))

    # Compute offsets per antenna and timestep
    time_ant_angles = ant_angles[np.newaxis,:]+time_angle[:,np.newaxis]

    A = np.empty(context.shape, context.dtype)
    U, V, W = A[:,:,0], A[:,:,1], A[:,:,2]
    U[:] = distance*np.cos(time_ant_angles)
    V[:] = distance*np.sin(time_ant_angles)
    W[:] = np.random.random(size=(ntime,na))*0.1

    # All antenna zero coordinates are set to (0,0,0)
    A[:,0,:] = 0

    return A

def identity_on_pols(self, context):
    """
    Returns [[1, 0], tiled up to other dimensions
             [0, 1]]
    """
    A = np.empty(context.shape, context.dtype)
    A[:,:,:] = [[[1,0,0,1]]]
    return A

def default_stokes(self, context):
    """
    Returns [[1, 0], tiled up to other dimensions
             [0, 0]]
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
    if np.product(context.shape) == 0:
        return np.empty(context.shape, context.dtype)

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
SECONDS = "Seconds"
DIMENSIONLESS = "Dimensionless"

LM_DESCRIPTION = ("(l,m) coordinates for {st} sources. "
            "Offset relative to the phase centre.")
STOKES_DESCRIPTION = ("(I,Q,U,V) Stokes parameters.")
ALPHA_DESCRIPTION = ("Power term describing the distribution of a source's flux "
            "over frequency. Distribution is calculated as (nu/nu_ref)^alpha "
            "where nu is frequency.")
REF_FREQ_DESCRIPTION = ("Reference frequency for {st} sources.")

# Tag Description
#
# input: arrays that must be input
# output: arrays that must be output
# temporary: arrays used to hold temporary results
# constant:

# List of arrays
A = [

    # Antenna Positions
    array_dict('antenna_position', ('na', 3), 'ft',
        default     = lambda s, c: np.zeros(c.shape, c.dtype),
        test        = lambda s, c: np.ones(c.shape, c.dtype),
        tags        = "input",
        description = "Antenna coordinates",
        units       = METERS),

    # Timesteps
    array_dict('time', ('ntime',), 'ft',
        default     = lambda s, c: np.linspace(0, 100,
                                    c.shape[0], dtype=c.dtype),
        test        = lambda s, c: np.linspace(0, 100,
                                    c.shape[0], dtype=c.dtype),
        tags        = "input",
        description = "Timesteps",
        units       = SECONDS),

    # Phase Centre
    array_dict('phase_centre', (2,), 'ft',
        default     = lambda s, c: np.array([0,0], dtype=c.dtype),
        test        = lambda s, c: np.array([0,0], dtype=c.dtype),
        tags        = "input",
        description = "Phase Centre",
        units       = RADIANS),

    # UVW Coordinates
    array_dict('uvw', ('ntime', 'na', 3), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = rand_uvw,
        tags    = "input, constant",
        description = "UVW antenna coordinates, normalised "
            "relative to a reference antenna.",
        units   = METERS),

    array_dict('antenna1', ('ntime', 'nbl'), np.int32,
        default = default_antenna1,
        test    = default_antenna1,
        tags    = "input",
        description = "Index of the first antenna "
            "of the baseline pair."),

    array_dict('antenna2', ('ntime', 'nbl'), np.int32,
        default = default_antenna2,
        test    = default_antenna2,
        tags    = "input",
        description = "Index of the second antenna "
            "of the baseline pair."),

    # Frequency
    array_dict('frequency', ('nchan',), 'ft',
        default = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], dtype=c.dtype),
        test    = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], dtype=c.dtype),
        tags    = "input",
        description = "Frequency. Frequencies from multiple bands "
            "are stacked on top of each other. ",
        units   = HERTZ),

    # Holographic Beam

    # Pointing errors
    array_dict('pointing_errors', ('ntime','na','nchan', 2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-2,
        tags    = "input, constant",
        description = "Pointing errors for each antenna. "
            "The components express an offset in the (l,m) plane.",
        units   = RADIANS),

    # Antenna scaling factors
    array_dict('antenna_scaling', ('na','nchan',2), 'ft',
        default = lambda s, c: np.ones(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype),
        tags    = "input, constant",
        description = "Antenna scaling factors for each antenna. "
            "The components express a scale in the (l,m) plane.",
        units   = DIMENSIONLESS),

    # Parallactic angles at each timestep for each antenna
    array_dict('parallactic_angles', ('ntime', 'na'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*np.pi,
        tags    = "input, constant",
        description = "Parallactic angles for each antenna.",
        units   = RADIANS),

    # Extents of the beam.
    # First 3 values are lower coordinate for (l, m, frequency)
    # while the last 3 are the upper coordinates
    array_dict('beam_extents', (6,), 'ft',
        default = lambda s, c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high]),
        test    = lambda s, c: c.dtype([-1, -1, _freq_low, 1, 1, _freq_high]),
        tags    = "input",
        description = "Extents of the holographic beam cube. "
            "[l_low, m_low, freq_low, l_high, m_high, freq_high].",
        units   = "[{r}, {r}, {h}, {r}, {r}, {h}]".format(r=RADIANS, h=HERTZ)),

    array_dict('beam_freq_map', ('beam_nud',), 'ft',
        default  = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True, dtype=c.dtype),
        test     = lambda s, c: np.linspace(_freq_low, _freq_high,
                                c.shape[0], endpoint=True, dtype=c.dtype),
        tags    = "input",
        description = "A map describing the frequency associated with each "
            "slice of the frequency dimension of the holographic beam cube.",
        units   = HERTZ),

    # Beam cube
    array_dict('ebeam', ('beam_lw', 'beam_mh', 'beam_nud', 'npol'), 'ct',
        default = identity_on_pols,
        test    = lambda s, c: rc(c.shape, c.dtype),
        tags    = "input, constant",
        description = "Holographic beam cube providing "
            "a discretised representation of the antenna beam pattern. "
            "Used to simulate the Direction Dependent Effects (DDE) "
            " or E term of the RIME."
            "Composed of a frequency stack of (l,m) images.",
        units   = DIMENSIONLESS),

    # Direction-Independent Effects
    array_dict('direction_independent_effects', ('ntime', 'na', 'nchan', 'npol'), 'ct',
        default = identity_on_pols,
        test    = lambda s, c: rc(c.shape, c.dtype),
        tags    = "input, constant",
        description = "Array providing the Direction Independent Effects (DIE) "
            "or G term of the RIME, term for each antenna.",
        units   = DIMENSIONLESS),

    # Point Source Definitions
    array_dict('point_lm', ('npsrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        tags    = "input, constant",
        description = LM_DESCRIPTION.format(st="point"),
        units   = RADIANS),
    array_dict('point_stokes', ('npsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        tags    = "input, constant",
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('point_alpha', ('npsrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        tags    = "input, constant",
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),
    array_dict('point_ref_freq', ('npsrc',), 'ft',
        default = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        test    = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        tags    = "input, constant",
        description = REF_FREQ_DESCRIPTION.format(st="point"),
        units   = HERTZ),

    # Gaussian Source Definitions
    array_dict('gaussian_lm', ('ngsrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        tags    = "input, constant",
        description = LM_DESCRIPTION.format(st="gaussian"),
        units   = RADIANS),
    array_dict('gaussian_stokes', ('ngsrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        tags    = "input, constant",
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('gaussian_alpha', ('ngsrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        tags    = "input, constant",
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),
    array_dict('gaussian_ref_freq', ('ngsrc',), 'ft',
        default = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        test    = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        tags    = "input, constant",
        description = REF_FREQ_DESCRIPTION.format(st="gaussian"),
        units   = HERTZ),
    array_dict('gaussian_shape', (3, 'ngsrc'), 'ft',
        default = default_gaussian_shape,
        test    = rand_gaussian_shape,
        tags    = "input, constant",
        description = "Parameters describing the shape of a gaussian source. "
            "(lproj, mproj, ratio) where lproj and mproj are the projections of "
            "the major and minor ellipse axes onto the l and m axes. "
            "Ratio is ratio of the minor/major axes.",
        units   = "({r}, {r}, {d})".format(r=RADIANS, d=DIMENSIONLESS)),

    # Sersic Source Definitions
    array_dict('sersic_lm', ('nssrc',2), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: (rf(c.shape, c.dtype)-0.5)*1e-1,
        tags    = "input, constant",
        description = LM_DESCRIPTION.format(st="sersic"),
        units   = "Radians"),
    array_dict('sersic_stokes', ('nssrc','ntime', 4), 'ft',
        default = default_stokes,
        test    = rand_stokes,
        tags    = "input, constant",
        description = STOKES_DESCRIPTION,
        units   = JANSKYS),
    array_dict('sersic_alpha', ('nssrc','ntime'), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype)*0.1,
        tags    = "input, constant",
        description = ALPHA_DESCRIPTION,
        units   = DIMENSIONLESS),
    array_dict('sersic_ref_freq', ('nssrc',), 'ft',
        default = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        test    = lambda s, c: np.full(c.shape, _ref_freq, c.dtype),
        tags    = "input, constant",
        description = REF_FREQ_DESCRIPTION.format(st="sersic"),
        units   = HERTZ),
    array_dict('sersic_shape', (3, 'nssrc'), 'ft',
        default = default_sersic_shape,
        test    = test_sersic_shape,
        tags    = "input, constant",
        description = "Parameters describing the shape of a sersic source. "
            "(e1, e2, eR). Further information is required here.",
        units   = "({r}, {r}, {d})".format(r=RADIANS, d=DIMENSIONLESS)),

    # Observation Data

    # Visibility flagging array
    array_dict('flag', ('ntime', 'nbl', 'nchan', 'npol'), np.uint8,
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: np.random.random_integers(0, 1,
            size=c.shape).astype(np.uint8),
        tags    = "input, constant",
        description = "Indicates whether a visibility should be flagged when "
            "computing a Residual or Chi-Squared value.",
        unut    = DIMENSIONLESS),
    # Weight array
    array_dict('weight', ('ntime','nbl','nchan', 'npol'), 'ft',
        default = lambda s, c: np.ones(c.shape, c.dtype),
        test    = lambda s, c: rf(c.shape, c.dtype),
        tags    = "input, constant",
        description = "Weight applied to the difference of observed and model "
            "visibilities when computing a Chi-Squared value.",
        units   = DIMENSIONLESS),
    # Observed Visibilities
    array_dict('observed_vis', ('ntime','nbl','nchan', 'npol'), 'ct',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rc(c.shape, c.dtype),
        tags    = "input, constant",
        description = "Observed visibilities, used to compute residuals and "
            "Chi-Squared values."),

    # Model Visibilities
    array_dict('model_vis', ('ntime','nbl','nchan', 'npol'), 'ct',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: rc(c.shape, c.dtype),
        tags    = ("input, output, constant"),
        description = "Model visibilities. In the context of input, these values "
            "will be added to the model visibilities computed by the RIME. "
            "This mechanism allows visibilities to be accumulated over different "
            "models, for example. However, they are zeroed by default."
            "In the context of output, these are the RIME model visibilities "),

    # Chi-squared
    array_dict('chi_squared', (1,), 'ft',
        default = lambda s, c: np.zeros(c.shape, c.dtype),
        test    = lambda s, c: np.zeros(c.shape, c.dtype),
        tags    = "output",
        description = "Chi-squared value associated with a tile "
            "of model and observed visibilities. These can be summed "
            "to produce a chi-squared for the entire problem."),

    # Result arrays
    array_dict('bsqrt', ('nsrc', 'ntime', 'nchan', 'npol'), 'ct',
        tags="temporary"),
    array_dict('cplx_phase', ('nsrc','ntime','na','nchan'), 'ct',
        tags="temporary"),
    array_dict('ejones', ('nsrc','ntime','na','nchan', 'npol'), 'ct',
        tags="temporary"),
    array_dict('ant_jones', ('nsrc','ntime','na','nchan', 'npol'), 'ct',
        tags="temporary"),
    array_dict('sgn_brightness', ('nsrc', 'ntime'), np.int8,
        tags="temporary"),
    array_dict('source_shape', ('nsrc', 'ntime', 'nbl', 'nchan'), 'ft',
        tags="temporary"),
    array_dict('chi_sqrd_result', ('ntime','nbl','nchan'), 'ft',
        tags="temporary"),
]
