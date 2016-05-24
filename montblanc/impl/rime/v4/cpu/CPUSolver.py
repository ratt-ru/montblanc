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

import numexpr as ne
import numpy as np

import montblanc
import montblanc.util as mbu
from montblanc.solvers import MontblancNumpySolver
from montblanc.config import RimeSolverConfig as Options

class CPUSolver(MontblancNumpySolver):
    def __init__(self, slvr_cfg):
        super(CPUSolver, self).__init__(slvr_cfg)

        # Monkey patch these functions onto the object
        # TODO: Remove this when deprecating v2.
        from montblanc.impl.rime.v4.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        from montblanc.impl.rime.v4.config import (A, P)

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E cube nu depth')

        self.register_properties(P)
        self.register_arrays(A)
        self.create_arrays()

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (ngsrc, ntime, nbl, nchan) matrix of floating point scalars.
        """

        ntime, nbl, ngsrc = self.dim_local_size('ntime', 'nbl', 'ngsrc')
        ant0, ant1 = self.ap_idx()

        # Calculate per baseline u from per antenna u
        up, uq = self.uvw[:,:,0][ant0], self.uvw[:,:,0][ant1]
        u = ne.evaluate('uq-up', {'up': up, 'uq': uq})

        # Calculate per baseline v from per antenna v
        vp, vq = self.uvw[:,:,1][ant0], self.uvw[:,:,1][ant1]
        v = ne.evaluate('vq-vp', {'vp': vp, 'vq': vq})

        # Calculate per baseline w from per antenna w
        wp, wq = self.uvw[:,:,2][ant0], self.uvw[:,:,2][ant1]
        w = ne.evaluate('wq-wp', {'wp': wp, 'wq': wq})

        el = self.gauss_shape[0]
        em = self.gauss_shape[1]
        R = self.gauss_shape[2]

        # OK, try obtain the same results with the fwhm factored out!
        # u1 = u*em - v*el
        # v1 = u*el + v*em
        u1 = ne.evaluate('u_em - v_el',
            {'u_em': np.outer(em, u), 'v_el': np.outer(el, v)})\
            .reshape(ngsrc, ntime,nbl)
        v1 = ne.evaluate('u_el + v_em', {
            'u_el' : np.outer(el, u), 'v_em' : np.outer(em, v)})\
            .reshape(ngsrc, ntime,nbl)

        scale_uv = (self.gauss_scale*self.frequency)\
            [np.newaxis,np.newaxis,np.newaxis,:]

        return ne.evaluate('exp(-((u1*scale_uv*R)**2 + (v1*scale_uv)**2))',
            local_dict={
                'u1':u1[:,:,:,np.newaxis],
                'v1':v1[:,:,:,np.newaxis],
                'scale_uv': scale_uv,
                'R':R[:,np.newaxis,np.newaxis,np.newaxis]})

    def compute_sersic_shape(self):
        """
        Compute the shape values for the sersic (exponential) sources.

        Returns a (nssrc, ntime, nbl, nchan) matrix of floating point scalars.
        """

        
        nssrc, ntime, nbl, nchan  = self.dim_local_size('nssrc', 'ntime', 'nbl', 'nchan')
        ant0, ant1 = self.ap_idx()

        # Calculate per baseline u from per antenna u
        up, uq = self.uvw[:,:,0][ant0], self.uvw[:,:,0][ant1]
        u = ne.evaluate('uq-up', {'up': up, 'uq': uq})

        # Calculate per baseline v from per antenna v
        vp, vq = self.uvw[:,:,1][ant0], self.uvw[:,:,1][ant1]
        v = ne.evaluate('vq-vp', {'vp': vp, 'vq': vq})

        # Calculate per baseline w from per antenna w
        wp, wq = self.uvw[:,:,2][ant0], self.uvw[:,:,2][ant1]
        w = ne.evaluate('wq-wp', {'wp': wp, 'wq': wq})

        e1 = self.sersic_shape[0]
        e2 = self.sersic_shape[1]
        R = self.sersic_shape[2]

        # OK, try obtain the same results with the fwhm factored out!
        # u1 = u*(1+e1) - v*e2
        # v1 = u*e2 + v*(1-e1)
        u1 = ne.evaluate('u_1_e1 + v_e2',
            {'u_1_e1': np.outer(np.ones(nssrc)+e1, u), 'v_e2' : np.outer(e2, v)})\
            .reshape(nssrc, ntime,nbl)
        v1 = ne.evaluate('u_e2 + v_1_e1', {
            'u_e2' : np.outer(e2, u), 'v_1_e1' : np.outer(np.ones(nssrc)-e1,v)})\
            .reshape(nssrc, ntime,nbl)

        # Obvious given the above reshape
        assert u1.shape == (nssrc, ntime, nbl)
        assert v1.shape == (nssrc, ntime, nbl)

        scale_uv = (self.two_pi_over_c * self.frequency)\
            [np.newaxis, np.newaxis, np.newaxis, :]

        den = ne.evaluate('1 + (u1*scale_uv*R)**2 + (v1*scale_uv*R)**2',
            local_dict={
                'u1': u1[:, :, :, np.newaxis],
                'v1': v1[:, :, :, np.newaxis],
                'scale_uv': scale_uv,
                'R': (R / (1 - e1 * e1 - e2 * e2))
                    [:,np.newaxis,np.newaxis,np.newaxis]})

        assert den.shape == (nssrc, ntime, nbl, nchan)

        return ne.evaluate('1/(den*sqrt(den))',
            { 'den' : den[:, :, :, :] })

    def compute_k_jones_scalar_per_ant(self):
        """
        Computes the scalar K (phase) term of the RIME per antenna.

        Returns a (nsrc,ntime,na,nchan) matrix of complex scalars.
        """
        
        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')

        freq = self.frequency

        u, v, w = self.uvw[:,:,0], self.uvw[:,:,1], self.uvw[:,:,2]
        l, m = self.lm[:,0], self.lm[:,1]

        # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
        n = ne.evaluate('sqrt(1. - l**2 - m**2) - 1.',
            {'l': l, 'm': m})

        # w*n+v*m+u*l. Outer product creates array of dim nsrcs x ntime x na
        phase = (np.outer(n, w) + np.outer(m, v) + np.outer(l, u)) \
                .reshape(nsrc, ntime, na)

        # e^(2*pi*sqrt(u*l+v*m+w*n)*frequency/C).
        # Dim. ntime x na x nchan x nsrcs
        cplx_phase = ne.evaluate('exp(-2*pi*1j*p*f/C)', {
            'p': phase[:, :, :, np.newaxis],
            'f': freq[np.newaxis, np.newaxis, np.newaxis, :],
            'C': montblanc.constants.C,
            'pi': np.pi
        })

        assert cplx_phase.shape == (nsrc, ntime, na, nchan)

        return cplx_phase

    def compute_kb_jones_per_ant(self):
        """
        Computes the K (phase) term, multiplied by the
        brightness matrix

        Returns a (4,nsrc,ntime,na,nchan) matrix of complex scalars.
        """

        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')

        k_jones = self.compute_k_jones_scalar_per_ant()
        assert k_jones.shape == (nsrc, ntime, na, nchan)
        b_jones = self.compute_b_jones()
        assert b_jones.shape == (4, nsrc, ntime, nchan)

        result = k_jones[np.newaxis,:,:,:,:]*b_jones[:,:,:,np.newaxis,:]
        assert result.shape == (4, nsrc, ntime, na, nchan)

        return result

    def compute_kb_sqrt_jones_per_ant(self):
        """
        Computes the K (phase) term, multiplied by the
        square root of the brightness matrix

        Returns a (nsrc,ntime,na,nchan,4) matrix of complex scalars.
        """

        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')

        k_jones = self.compute_k_jones_scalar_per_ant()
        b_sqrt_jones = self.compute_b_sqrt_jones()

        result = k_jones[:,:,:,:,np.newaxis]*b_sqrt_jones[:,:,np.newaxis,:,:]
        assert result.shape == (nsrc, ntime, na, nchan, 4)

        return result

    def compute_b_jones(self):
        """
        Computes the brightness matrix from the stokes parameters.

        Returns a (4,nsrc,ntime,nchan) matrix of complex scalars.
        """
        nsrc, ntime, nchan = self.dim_local_size('nsrc', 'ntime', 'nchan')

        try:
            B = np.empty(shape=(nsrc, ntime, 4), dtype=self.ct)
            S = self.stokes
            # Create the brightness matrix from the stokes parameters
            # Dimension (nsrc, ntime, 4)
            B[:,:,0] = S[:,:,0] + S[:,:,1]    # I+Q
            B[:,:,1] = S[:,:,2] + 1j*S[:,:,3] # U+Vi
            B[:,:,2] = S[:,:,2] - 1j*S[:,:,3] # U-Vi
            B[:,:,3] = S[:,:,0] - S[:,:,1]    # I-Q

            # Multiply the scalar power term into the matrix
            B_power = ne.evaluate('B*((f/rf)**a)', {
                 'rf': self.ref_freq,
                 'B': B[:,:,np.newaxis,:],
                 'f': self.frequency[np.newaxis, np.newaxis, :, np.newaxis],
                 'a': self.alpha[:, :, np.newaxis, np.newaxis] })

            assert B_power.shape == (nsrc, ntime, nchan, 4)

            return B_power

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_b_sqrt_jones(self, b_jones=None):
        """
        Computes the square root of the brightness matrix.

        Returns a (4,nsrc,ntime,nchan) matrix of complex scalars.
        """
        nsrc, ntime, nchan = self.dim_local_size('nsrc', 'ntime', 'nchan')

        try:
            # See
            # http://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
            # Note that this code handles a special case of the above
            # where we assume that both the trace and determinant
            # are real and positive.
            B = self.compute_b_jones() if b_jones is None else b_jones.copy()

            # trace = I+Q + I-Q = 2*I
            # det = (I+Q)*(I-Q) - (U+iV)*(U-iV) = I**2-Q**2-U**2-V**2
            trace = (B[:,:,:,0]+B[:,:,:,3]).real
            det = (B[:,:,:,0]*B[:,:,:,3] - B[:,:,:,1]*B[:,:,:,2]).real

            assert trace.shape == (nsrc, ntime, nchan)
            assert det.shape == (nsrc, ntime, nchan)

            assert np.all(trace >= 0.0), \
                'Negative brightness matrix trace'
            assert np.all(det >= 0.0), \
                'Negative brightness matrix determinant'

            s = np.sqrt(det)
            t = np.sqrt(trace + 2*s)

            # We don't have a solution for matrices
            # where both s and t are zero. In the case
            # of brightness matrices, zero s and t
            # implies that the matrix itself is 0.
            # Avoid infs and nans from divide by zero
            mask = np.logical_and(s == 0.0, t == 0.0)
            t[mask] = 1.0

            # Add s to the diagonal entries
            B[:,:,:,0] += s
            B[:,:,:,3] += s

            # Divide the entire matrix by t
            B /= t[:,:,:,np.newaxis]

            return B

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def trilinear_interpolate(self, sum, abs_sum,
            gl, gm, gchan, weight):
        """
        Given a grid position in the beam cube (gl, gm, gchan),
        and positive unit offsets from this position (ld, lm, chd),
        round the grid position to integer positions and add the offset.
        Then, load in the complex number at the computed grid position,
        weight it with the distance from the original grid position,
        and add it to the sum and abs_sum arguments.
        """
        nsrc, ntime, na, nchan, beam_lw, beam_mh, beam_nud = (
            self.dim_local_size('nsrc', 'ntime', 'na', 'nchan',
                'beam_lw', 'beam_mh', 'beam_nud'))

        # Does the source lie within the beam cube?
        invalid_l = np.logical_or(gl < 0.0, gl >= beam_lw)
        invalid_m = np.logical_or(gm < 0.0, gm >= beam_mh)
        invalid_lm = np.logical_or.reduce((invalid_l, invalid_m))

        assert invalid_lm.shape == (nsrc, ntime, na, nchan)

        # Just set coordinates and weights to zero
        # if they're outside the cube
        gl[invalid_lm] = 0
        gm[invalid_lm] = 0
        weight[invalid_lm] = 0

        # Indices within the cube
        l_idx = gl.astype(np.int32)
        m_idx = gm.astype(np.int32)
        ch_idx = gchan.astype(np.int32)[np.newaxis,np.newaxis,np.newaxis,:]

        beam_pols = self.E_beam[l_idx,m_idx,ch_idx]
        assert beam_pols.shape == (nsrc, ntime, na, nchan, 4)

        sum += weight[:,:,:,:,np.newaxis]*beam_pols
        abs_sum += weight[:,:,:,:,np.newaxis]*np.abs(beam_pols)

    def compute_E_beam(self):
        """
        Rotates sources through a beam cube. At each timestep,
        the source position is computed within the grid defining
        the cube, taking into account pointing errors and
        scaling parameters for each antenna.

        The complex numbers at the eight grid points surrounding
        the source are bilinearly interpolated together to
        produce a single complex number, the
        Direction-Dependent Effect for the source at a particular
        time, antenna and frequency.

        Returns a (nsrc,ntime,na,nchan,4) matrix of complex scalars.
        """
        nsrc, ntime, na, nchan, beam_lw, beam_mh, beam_nud = (
            self.dim_local_size('nsrc', 'ntime', 'na', 'nchan',
                'beam_lw', 'beam_mh', 'beam_nud'))

        sint = np.sin(self.parallactic_angle*np.arange(ntime))
        cost = np.cos(self.parallactic_angle*np.arange(ntime))

        assert sint.shape == (ntime,)
        assert cost.shape == (ntime,)

        l0, m0 = self.lm[:,0], self.lm[:,1]
        l = l0[:,np.newaxis]*cost[np.newaxis,:] - m0[:,np.newaxis]*sint[np.newaxis,:]
        m = l0[:,np.newaxis]*sint[np.newaxis,:] + m0[:,np.newaxis]*cost[np.newaxis,:]

        assert l.shape == (nsrc, ntime)
        assert m.shape == (nsrc, ntime)

        ld, md = self.point_errors[:,:,:,0], self.point_errors[:,:,:,1]
        l = l[:,:,np.newaxis,np.newaxis] + ld[np.newaxis,:,:,:]
        m = m[:,:,np.newaxis,np.newaxis] + md[np.newaxis,:,:,:]

        assert l.shape == (nsrc, ntime, na, nchan)
        assert m.shape == (nsrc, ntime, na, nchan)

        a, b = self.antenna_scaling[:,:,0], self.antenna_scaling[:,:,1]
        l *= a[np.newaxis, np.newaxis, :, :]
        m *= b[np.newaxis, np.newaxis, :, :]

        # Compute grid position and difference from
        # actual position for the source at each channel
        l = (beam_lw-1) * (l-self.beam_ll) / (self.beam_ul-self.beam_ll)
        assert l.shape == (nsrc, ntime, na, nchan)
        gl = np.floor(l)
        ld = l - gl

        m = (beam_mh-1) * (m-self.beam_lm) / (self.beam_um-self.beam_lm)
        assert m.shape == (nsrc, ntime, na, nchan)
        gm = np.floor(m)
        md = m - gm

        # Work out where we are in the beam cube, relative
        # the position in the global channel space.
        # Get our problem extents and the global size of
        # the channel dimension
        chan_low, chan_high = self.dim_extents('nchan')
        nchan_global = self.dim_global_size('nchan')

        # Global channel size is divisor, but handle one channel case
        div = self.ft(nchan_global-1) if nchan_global > 1 else 1.0
        # Create the channel range from the extents
        chan_range = np.arange(chan_low, chan_high).astype(self.ft)
        # Divide channel range by global size and multiply
        # to obtain position in the beam cube
        chan = (beam_nud-1)*chan_range / div
        assert chan.shape == (chan_high - chan_low, )
        gchan = np.floor(chan)
        chd = (chan - gchan)[np.newaxis,np.newaxis,np.newaxis,:]

        # Handle the boundary case where the channel
        # lies on the last grid point
        fiddle = (chan == beam_nud - 1)
        gchan[fiddle] = beam_nud - 2
        chd[:,:,:,fiddle] = 1

        # Initialise the sum to zero
        sum = np.zeros_like(self.jones)
        abs_sum = np.zeros(shape=sum.shape, dtype=self.ft)

        # A simplified bilinear weighting is used here. Given
        # point x between points x1 and x2, with function f
        # provided values f(x1) and f(x2) at these points.
        #
        # x1 ------- x ---------- x2
        #
        # Then, the value of f can be approximated using the following:
        # f(x) ~= f(x1)(x2-x)/(x2-x1) + f(x2)(x-x1)/(x2-x1)
        #
        # Note how the value f(x1) is weighted with the distance
        # from the opposite point (x2-x).
        #
        # As we are interpolating on a grid, we have the following
        # 1. (x2 - x1) == 1
        # 2. (x - x1)  == 1 - 1 + (x - x1)
        #              == 1 - (x2 - x1) + (x - x1)
        #              == 1 - (x2 - x)
        # 2. (x2 - x)  == 1 - 1 + (x2 - x)
        #              == 1 - (x2 - x1) + (x2 - x)
        #              == 1 - (x - x1)
        #
        # Extending the above to 3D, we have
        # f(x,y,z) ~= f(x1,y1,z1)(x2-x)(y2-y)(z2-z) + ...
        #           + f(x2,y2,z2)(x-x1)(y-y1)(z-z1)
        #
        # f(x,y,z) ~= f(x1,y1,z1)(1-(x-x1))(1-(y-y1))(1-(z-z1)) + ...
        #           + f(x2,y2,z2)   (x-x1)    (y-y1)    (z-z1)

        # Load in the complex values from the E beam
        # at the supplied coordinate offsets.
        # Save the sum of abs in sum.real
        # and the sum of args in sum.imag
        self.trilinear_interpolate(sum, abs_sum,
            gl + 0, gm + 0, gchan + 0, (1-ld)*(1-md)*(1-chd))
        self.trilinear_interpolate(sum, abs_sum,
            gl + 1, gm + 0, gchan + 0, ld*(1-md)*(1-chd))
        self.trilinear_interpolate(sum, abs_sum,
            gl + 0, gm + 1, gchan + 0, (1-ld)*md*(1-chd))
        self.trilinear_interpolate(sum, abs_sum,
            gl + 1, gm + 1, gchan + 0, ld*md*(1-chd))

        self.trilinear_interpolate(sum, abs_sum,
            gl + 0, gm + 0, gchan + 1, (1-ld)*(1-md)*chd)
        self.trilinear_interpolate(sum, abs_sum,
            gl + 1, gm + 0, gchan + 1, ld*(1-md)*chd)
        self.trilinear_interpolate(sum, abs_sum,
            gl + 0, gm + 1, gchan + 1, (1-ld)*md*chd)
        self.trilinear_interpolate(sum, abs_sum,
            gl + 1, gm + 1, gchan + 1, ld*md*chd)

        # Determine the angle of the polarisation
        angle = np.angle(sum)

        assert angle.shape == (nsrc, ntime, na, nchan, 4)
        assert abs_sum.shape == (nsrc, ntime, na, nchan, 4)

        # Take the complex exponent of the angle
        # and multiply by the sum of abs
        return abs_sum*np.exp(1j*angle)

    @staticmethod
    def jones_multiply(A, B, hermitian=None, jones_shape=None):
        if hermitian is None:
            hermitian = False

        if jones_shape is None:
            jones_shape = '1x4'

        if type(hermitian) != type(True):
            raise ValueError('hermitian must be True or False')

        if hermitian:
            result = np.einsum("...ij,...kj->...ik",
                A.reshape(-1,2,2), B.reshape(-1,2,2).conj())
        else:
            result = np.einsum("...ij,...jk->...ik",
                A.reshape(-1,2,2), B.reshape(-1,2,2))

        if jones_shape == '1x4':
            return result.reshape(-1, 4)
        elif jones_shape == '2x2':
            return result
        else:
            raise ValueError("jones_shape must be '1x4' or '2x2'.")

    def compute_ekb_sqrt_jones_per_ant(self):
        """
        Computes the per antenna jones matrices, the product
        of E x K x B_sqrt

        Returns a (nsrc,ntime,na,nchan,4) matrix of complex scalars.
        """
        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')
        N = nsrc*ntime*na*nchan

        E_beam = self.compute_E_beam()
        kb_sqrt = self.compute_kb_sqrt_jones_per_ant()

        assert E_beam.shape == (nsrc, ntime, na, nchan, 4)
        assert kb_sqrt.shape == (nsrc, ntime, na, nchan, 4)

        result = CPUSolver.jones_multiply(E_beam, kb_sqrt)
        return result.reshape(nsrc, ntime, na, nchan, 4)

    def compute_ekb_jones_per_bl(self, ekb_sqrt=None):
        """
        Computes per baseline jones matrices based on the
        scalar EKB Square root terms

        Returns a (nsrc,ntime,nbl,nchan,4) matrix of complex scalars.
        """
        nsrc, npsrc, ngsrc, nssrc, ntime, nbl, nchan = self.dim_local_size(
            'nsrc', 'npsrc', 'ngsrc', 'nssrc', 'ntime', 'nbl', 'nchan')

        try:
            if ekb_sqrt is None:
                ekb_sqrt = self.compute_ekb_sqrt_jones_per_ant()

            ant0, ant1 = self.ap_idx(src=True, chan=True)

            ekb_sqrt_p = ekb_sqrt[ant0]
            ekb_sqrt_q = ekb_sqrt[ant1]
            assert ekb_sqrt_p.shape == (nsrc, ntime, nbl, nchan, 4)

            result = self.jones_multiply(ekb_sqrt_p, ekb_sqrt_q,
                hermitian=True).reshape(nsrc, ntime, nbl, nchan, 4)

            # Multiply in Gaussian Shape Terms
            if ngsrc > 0:
                src_beg = npsrc
                src_end = npsrc + ngsrc
                gauss_shape = self.compute_gaussian_shape()
                result[src_beg:src_end,:,:,:,:] *= gauss_shape[:,:,:,:,np.newaxis]

            # Multiply in Sersic Shape Terms
            if nssrc > 0:
                src_beg = npsrc + ngsrc
                src_end = npsrc + ngsrc + nssrc
                sersic_shape = self.compute_sersic_shape()
                result[src_beg:src_end,:,:,:,:] *= sersic_shape[:,:,:,:,np.newaxis]

            return result
            #return ebk_sqrt[1]*ebk_sqrt[0].conj()
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_ekb_vis(self, ekb_jones=None):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (ntime,nbl,nchan,4) matrix of complex scalars.
        """
        nsrc, ntime, nbl, nchan = self.dim_local_size('nsrc', 'ntime', 'nbl', 'nchan')

        if ekb_jones is None:
            ekb_jones = self.compute_ekb_jones_per_bl()

        want_shape = (nsrc, ntime, nbl, nchan, 4)
        assert ekb_jones.shape == want_shape, \
            'Expected shape %s. Got %s instead.' % \
            (want_shape, ekb_jones.shape)

        if nsrc == 1:
            # Due to this bug
            # https://github.com/pydata/numexpr/issues/79
            # numexpr may not reduce a source axis of size 1
            # Work around this
            vis = ekb_jones.squeeze(0)
        else:
            vis = ne.evaluate('sum(ebk,0)',
                {'ebk': ekb_jones }) \
                .astype(self.ct)

        assert vis.shape == (ntime, nbl, nchan, 4)

        return vis

    def compute_gekb_vis(self, ekb_vis=None):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (ntime,nbl,nchan,4) matrix of complex scalars.
        """
        nsrc, ntime, nbl, nchan = self.dim_local_size('nsrc', 'ntime', 'nbl', 'nchan')

        if ekb_vis is None:
            ekb_vis = self.compute_ekb_vis()

        want_shape = (ntime, nbl, nchan, 4)
        assert ekb_vis.shape == want_shape, \
            'Expected shape %s. Got %s instead.' % \
            (want_shape, ekb_vis.shape)

        ant0, ant1 = self.ap_idx(chan=True)
        g_term_p = self.G_term[ant0]
        g_term_q = self.G_term[ant1]

        assert g_term_p.shape == (ntime, nbl, nchan, 4)

        result = (self.jones_multiply(g_term_p, ekb_vis)
            .reshape(ntime, nbl, nchan, 4))

        result = (self.jones_multiply(result, g_term_q, hermitian=True)
            .reshape(ntime, nbl, nchan, 4))

        # Output residuals if requested, otherwise return
        # visibilities after flagging
        if self.outputs_residuals():
            result = ne.evaluate('(ovis - mvis)*where(flag > 0, 0, 1)', {
                'mvis': result,
                'ovis': self.observed_vis,
                'flag' : self.flag })
            assert result.shape == (ntime, nbl, nchan, 4)
        else:
            result[self.flag > 0] = 0

        return result

    def compute_chi_sqrd_sum_terms(self, vis=None):
        """
        Computes the terms of the chi squared sum,
        but does not perform the sum itself.

        Returns a (ntime,nbl,nchan) matrix of floating point scalars.
        """
        ntime, nbl, nchan = self.dim_local_size('ntime', 'nbl', 'nchan')

        if vis is None:
            vis = self.compute_gekb_vis()

        # Compute the residuals if this has not yet happened
        if not self.outputs_residuals():
            d = ne.evaluate('(ovis - mvis)*where(flag > 0, 0, 1)', {
                'mvis': vis,
                'ovis': self.observed_vis,
                'flag' : self.flag })
            assert d.shape == (ntime, nbl, nchan, 4)
        else:
            d = vis

        # Square of the real and imaginary components
        re = ne.evaluate('re**2', {'re': d.real})
        im = ne.evaluate('im**2', {'im': d.imag})
        wv = self.weight_vector

        # Multiply by the weight vector if required
        if self.use_weight_vector() is True:
            ne.evaluate('re*wv', {'re': re, 'wv': wv}, out=re)
            ne.evaluate('im*wv', {'im': im, 'wv': wv}, out=im)

        # Reduces a dimension so that we have (nbl,nchan,ntime)
        # (XX.real^2 + XY.real^2 + YX.real^2 + YY.real^2) +
        # ((XX.imag^2 + XY.imag^2 + YX.imag^2 + YY.imag^2))

        # Sum the real and imaginary terms together
        # for the final result.
        re_sum = ne.evaluate('sum(re,3)', {'re': re})
        im_sum = ne.evaluate('sum(im,3)', {'im': im})
        chi_sqrd_terms = ne.evaluate('re_sum + im_sum',
            {'re_sum': re_sum, 'im_sum': im_sum})
        assert chi_sqrd_terms.shape == (ntime, nbl, nchan)

        return chi_sqrd_terms

    def compute_chi_sqrd(self, chi_sqrd_terms=None):
        """ Computes the floating point chi squared value. """

        if chi_sqrd_terms is None:
            chi_sqrd_terms = self.compute_chi_sqrd_sum_terms()

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum

        term_sum = ne.evaluate('sum(terms)', {'terms': chi_sqrd_terms})
        return (term_sum if self.use_weight_vector() is True
            else term_sum / self.sigma_sqrd)

    def solve(self):
        """ Solve the RIME """

        self.jones[:] = self.compute_ekb_sqrt_jones_per_ant()
        
        self.vis[:] = self.compute_gekb_vis()
        
        self.chi_sqrd_result[:] = self.compute_chi_sqrd_sum_terms(
            self.vis)
        
        self.set_X2(self.compute_chi_sqrd(self.chi_sqrd_result))