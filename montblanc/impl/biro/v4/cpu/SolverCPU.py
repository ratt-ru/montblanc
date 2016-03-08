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

class SolverCPU(object):
    def __init__(self, solver):
        self.solver = solver

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (ngsrc, ntime, nbl, nchan) matrix of floating point scalars.
        """

        slvr = self.solver

        try:
            ap = slvr.get_ap_idx()

            # Calculate per baseline u from per antenna u
            u = slvr.uvw_cpu[:,:,0][ap]
            u = ne.evaluate('ap-aq', {'ap': u[0], 'aq': u[1]})

            # Calculate per baseline v from per antenna v
            v = slvr.uvw_cpu[:,:,1][ap]
            v = ne.evaluate('ap-aq', {'ap': v[0], 'aq': v[1]})

            # Calculate per baseline w from per antenna w
            w = slvr.uvw_cpu[:,:,2][ap]
            w = ne.evaluate('ap-aq', {'ap': w[0], 'aq': w[1]})

            el = slvr.gauss_shape_cpu[0]
            em = slvr.gauss_shape_cpu[1]
            R = slvr.gauss_shape_cpu[2]

            # OK, try obtain the same results with the fwhm factored out!
            # u1 = u*em - v*el
            # v1 = u*el + v*em
            u1 = ne.evaluate('u_em - v_el',
                {'u_em': np.outer(em, u), 'v_el': np.outer(el, v)})\
                .reshape(slvr.ngsrc, slvr.ntime,slvr.nbl)
            v1 = ne.evaluate('u_el + v_em', {
                'u_el' : np.outer(el, u), 'v_em' : np.outer(em, v)})\
                .reshape(slvr.ngsrc, slvr.ntime,slvr.nbl)

            scale_uv = (slvr.gauss_scale*slvr.frequency_cpu)\
                [np.newaxis,np.newaxis,np.newaxis,:]

            return ne.evaluate('exp(-((u1*scale_uv*R)**2 + (v1*scale_uv)**2))',
                local_dict={
                    'u1':u1[:,:,:,np.newaxis],
                    'v1':v1[:,:,:,np.newaxis],
                    'scale_uv': scale_uv,
                    'R':R[:,np.newaxis,np.newaxis,np.newaxis]})

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_sersic_shape(self):
        """
        Compute the shape values for the sersic (exponential) sources.

        Returns a (nssrc, ntime, nbl, nchan) matrix of floating point scalars.
        """

        slvr = self.solver

        try:
            ap = slvr.get_ap_idx()

            # Calculate per baseline u from per antenna u
            u = slvr.uvw_cpu[:,:,0][ap]
            u = ne.evaluate('ap-aq', {'ap': u[0], 'aq': u[1]})

            # Calculate per baseline v from per antenna v
            v = slvr.uvw_cpu[:,:,1][ap]
            v = ne.evaluate('ap-aq', {'ap': v[0], 'aq': v[1]})

            # Calculate per baseline w from per antenna w
            w = slvr.uvw_cpu[:,:,2][ap]
            w = ne.evaluate('ap-aq', {'ap': w[0], 'aq': w[1]})

            e1 = slvr.sersic_shape_cpu[0]
            e2 = slvr.sersic_shape_cpu[1]
            R = slvr.sersic_shape_cpu[2]

            # OK, try obtain the same results with the fwhm factored out!
            # u1 = u*(1+e1) - v*e2
            # v1 = u*e2 + v*(1-e1)
            u1 = ne.evaluate('u_1_e1 + v_e2',
                {'u_1_e1': np.outer(np.ones(slvr.nssrc)+e1, u), 'v_e2' : np.outer(e2, v)})\
                .reshape(slvr.nssrc, slvr.ntime,slvr.nbl)
            v1 = ne.evaluate('u_e2 + v_1_e1', {
                'u_e2' : np.outer(e2, u), 'v_1_e1' : np.outer(np.ones(slvr.nssrc)-e1,v)})\
                .reshape(slvr.nssrc, slvr.ntime,slvr.nbl)

            # Obvious given the above reshape
            assert u1.shape == (slvr.nssrc, slvr.ntime, slvr.nbl)
            assert v1.shape == (slvr.nssrc, slvr.ntime, slvr.nbl)

            scale_uv = (slvr.two_pi_over_c * slvr.frequency_cpu)\
                [np.newaxis, np.newaxis, np.newaxis, :]

            den = ne.evaluate('1 + (u1*scale_uv*R)**2 + (v1*scale_uv*R)**2',
                local_dict={
                    'u1': u1[:, :, :, np.newaxis],
                    'v1': v1[:, :, :, np.newaxis],
                    'scale_uv': scale_uv,
                    'R': (R / (1 - e1 * e1 - e2 * e2))
                        [:,np.newaxis,np.newaxis,np.newaxis]})

            assert den.shape == (slvr.nssrc, slvr.ntime, slvr.nbl, slvr.nchan)

            return ne.evaluate('1/(den*sqrt(den))',
                { 'den' : den[:, :, :, :] })

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_k_jones_scalar_per_ant(self):
        """
        Computes the scalar K (phase) term of the RIME per antenna.

        Returns a (nsrc,ntime,na,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            freq = slvr.frequency_cpu

            u, v, w = slvr.uvw_cpu[:,:,0], slvr.uvw_cpu[:,:,1], slvr.uvw_cpu[:,:,2]
            l, m = slvr.lm_cpu[:,0], slvr.lm_cpu[:,1]

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
            n = ne.evaluate('sqrt(1. - l**2 - m**2) - 1.',
                {'l': l, 'm': m})

            # w*n+v*m+u*l. Outer product creates array of dim nsrcs x ntime x na
            phase = (np.outer(n, w) + np.outer(m, v) + np.outer(l, u)) \
                    .reshape(slvr.nsrc, slvr.ntime, slvr.na)

            # e^(2*pi*sqrt(u*l+v*m+w*n)*frequency/C).
            # Dim. ntime x na x nchan x nsrcs
            cplx_phase = ne.evaluate('exp(-2*pi*1j*p*f/C)', {
                'p': phase[:, :, :, np.newaxis],
                'f': freq[np.newaxis, np.newaxis, np.newaxis, :],
                'C': montblanc.constants.C,
                'pi': np.pi
            })

            assert cplx_phase.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

            return cplx_phase

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_kb_jones_per_ant(self):
        """
        Computes the K (phase) term, multiplied by the
        brightness matrix

        Returns a (4,nsrc,ntime,na,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        k_jones = self.compute_k_jones_scalar_per_ant()
        assert k_jones.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)
        b_jones = self.compute_b_jones()
        assert b_jones.shape == (4, slvr.nsrc, slvr.ntime, slvr.nchan)

        result = k_jones[np.newaxis,:,:,:,:]*b_jones[:,:,:,np.newaxis,:]
        assert result.shape == (4, slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

        return result

    def compute_kb_sqrt_jones_per_ant(self):
        """
        Computes the K (phase) term, multiplied by the
        square root of the brightness matrix

        Returns a (4,nsrc,ntime,na,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        k_jones = self.compute_k_jones_scalar_per_ant()
        b_sqrt_jones = self.compute_b_sqrt_jones()

        result = k_jones[np.newaxis,:,:,:,:]*b_sqrt_jones[:,:,:,np.newaxis,:]
        assert result.shape == (4, slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

        return result

    def compute_b_jones(self):
        """
        Computes the brightness matrix from the stokes parameters.

        Returns a (4,nsrc,ntime,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Create the brightness matrix. Dim 4 x nsrcs x ntime
            B = np.array([
                # fI+fQ + 0j
                slvr.stokes_cpu[:,:,0]+slvr.stokes_cpu[:,:,1] + 0j,
                # fU + fV*1j
                slvr.stokes_cpu[:,:,2] + 1j*slvr.stokes_cpu[:,:,3],
                # fU - fV*1j
                slvr.stokes_cpu[:,:,2] - 1j*slvr.stokes_cpu[:,:,3],
                # fI-fQ + 0j
                slvr.stokes_cpu[:,:,0]-slvr.stokes_cpu[:,:,1] + 0j],
                dtype=slvr.ct)

            assert B.shape == (4, slvr.nsrc, slvr.ntime)

            # Multiply the scalar power term into the matrix
            B_power = ne.evaluate('B*((f/rf)**a)', {
                 'rf': slvr.ref_freq,
                 'B': B[:,:,:,np.newaxis],
                 'f': slvr.frequency_cpu[np.newaxis, np.newaxis, np.newaxis, :],
                 'a': slvr.alpha_cpu[np.newaxis, :, :, np.newaxis] })

            assert B_power.shape == (4, slvr.nsrc, slvr.ntime, slvr.nchan)

            return B_power

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_b_sqrt_jones(self, b_jones=None):
        """
        Computes the square root of the brightness matrix.

        Returns a (4,nsrc,ntime,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # See
            # http://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
            # Note that this code handles a special case of the above
            # where we assume that both the trace and determinant
            # are real and positive.
            B = self.compute_b_jones() if b_jones is None else b_jones.copy()

            # trace = I+Q + I-Q = 2*I
            # det = (I+Q)*(I-Q) - (U+iV)*(U-iV) = I**2-Q**2-U**2-V**2
            trace = (B[0]+B[3]).real
            det = (B[0]*B[3] - B[1]*B[2]).real

            assert trace.shape == (slvr.nsrc, slvr.ntime, slvr.nchan)
            assert det.shape == (slvr.nsrc, slvr.ntime, slvr.nchan)

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
            B[0] += s
            B[3] += s

            # Divide the entire matrix by t
            B /= t

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
        slvr = self.solver

        # Does the source lie within the beam cube?
        invalid_l = np.logical_or(gl < 0.0, gl >= slvr.beam_lw)
        invalid_m = np.logical_or(gm < 0.0, gm >= slvr.beam_mh)
        invalid_lm = np.logical_or.reduce((invalid_l, invalid_m))

        assert invalid_lm.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

        # Just set coordinates and weights to zero
        # if they're outside the cube
        gl[invalid_lm] = 0
        gm[invalid_lm] = 0
        weight[invalid_lm] = 0

        # Indices within the cube
        l_idx = gl.astype(np.int32)
        m_idx = gm.astype(np.int32)
        ch_idx = gchan.astype(np.int32)[np.newaxis,np.newaxis,np.newaxis,:]

        beam_pols = slvr.E_beam_cpu[l_idx,m_idx,ch_idx]
        assert beam_pols.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)

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
        slvr = self.solver

        sint = np.sin(slvr.parallactic_angle*np.arange(slvr.ntime))
        cost = np.cos(slvr.parallactic_angle*np.arange(slvr.ntime))

        assert sint.shape == (slvr.ntime,)
        assert cost.shape == (slvr.ntime,)

        l0, m0 = slvr.lm_cpu[:,0], slvr.lm_cpu[:,1]
        l = l0[:,np.newaxis]*cost[np.newaxis,:] - m0[:,np.newaxis]*sint[np.newaxis,:]
        m = l0[:,np.newaxis]*sint[np.newaxis,:] + m0[:,np.newaxis]*cost[np.newaxis,:]

        assert l.shape == (slvr.nsrc, slvr.ntime)
        assert m.shape == (slvr.nsrc, slvr.ntime)

        ld, md = slvr.point_errors_cpu[:,:,:,0], slvr.point_errors_cpu[:,:,:,1]
        l = l[:,:,np.newaxis,np.newaxis] + ld[np.newaxis,:,:,:]
        m = m[:,:,np.newaxis,np.newaxis] + md[np.newaxis,:,:,:]

        assert l.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)
        assert m.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

        a, b = slvr.antenna_scaling_cpu[:,:,0], slvr.antenna_scaling_cpu[:,:,1]
        l *= a[np.newaxis, np.newaxis, :, :]
        m *= b[np.newaxis, np.newaxis, :, :]

        # Compute grid position and difference from
        # actual position for the source at each channel
        l = (slvr.beam_lw-1) * (l-slvr.beam_ll) / (slvr.beam_ul-slvr.beam_ll)
        assert l.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)
        gl = np.floor(l)
        ld = l - gl

        m = (slvr.beam_mh-1) * (m-slvr.beam_lm) / (slvr.beam_um-slvr.beam_lm)
        assert m.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)
        gm = np.floor(m)
        md = m - gm

        div = slvr.ft(slvr.nchan-1) if slvr.nchan > 1 else 1.0
        chan_range = np.arange(slvr.nchan).astype(slvr.ft)
        chan = (slvr.beam_nud-1)*chan_range / div
        assert chan.shape == (slvr.nchan, )
        gchan = np.floor(chan)
        chd = (chan - gchan)[np.newaxis,np.newaxis,np.newaxis,:]

        # Handle the boundary case where the channel
        # lies on the last grid point
        fiddle = (chan == slvr.beam_nud - 1)
        gchan[fiddle] = slvr.beam_nud - 2
        chd[:,:,:,fiddle] = 1

        # Initialise the sum to zero
        sum = np.zeros_like(slvr.jones_cpu)
        abs_sum = np.zeros(shape=sum.shape, dtype=slvr.ft)

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

        assert angle.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)
        assert abs_sum.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)

        # Take the complex exponent of the angle
        # and multiply by the sum of abs
        return abs_sum*np.exp(1j*angle)

    @staticmethod
    def jones_multiply(A, B, N):
        # Based on
        # https://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/

        AR = A.reshape(N, 2, 2)
        #b = B.reshape(N, 2, 2)

        return np.sum(
            AR.transpose(0,2,1).reshape(N,2,2,1)*B.reshape(N,2,1,2),
            -3)

    @staticmethod
    def jones_multiply_hermitian_transpose(A, B, N):
        # Based on
        # https://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/

        AR = A.reshape(N, 2, 2)
        BRT = B.reshape(N, 2, 2).transpose(0,2,1).conj()
        #b = B.reshape(N, 2, 2)

        result = np.sum(
            AR.transpose(0,2,1).reshape(N,2,2,1)*BRT.reshape(N,2,1,2),
            -3)

        return result

    def compute_ekb_sqrt_jones_per_ant(self):
        """
        Computes the per antenna jones matrices, the product
        of E x K x B_sqrt

        Returns a (nsrc,ntime,na,nchan,4) matrix of complex scalars.
        """

        slvr = self.solver
        N = slvr.nsrc*slvr.ntime*slvr.na*slvr.nchan

        E_beam = self.compute_E_beam()
        kb_sqrt = self.compute_kb_sqrt_jones_per_ant().transpose(1,2,3,4,0)

        assert E_beam.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)
        assert kb_sqrt.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)

        result = SolverCPU.jones_multiply(E_beam, kb_sqrt,
            slvr.nsrc*slvr.ntime*slvr.na*slvr.nchan)
        return result.reshape(slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan, 4)

    def compute_ekb_jones_per_bl(self, ekb_sqrt=None):
        """
        Computes per baseline jones matrices based on the
        scalar EKB Square root terms

        Returns a (nsrc,ntime,nbl,nchan,4) matrix of complex scalars.
        """

        slvr = self.solver

        try:
            if ekb_sqrt is None:
                ekb_sqrt = self.compute_ekb_sqrt_jones_per_ant()

            ap = slvr.get_ap_idx(src=True, chan=True)
            ekb_sqrt_idx = ekb_sqrt[ap]
            assert ekb_sqrt_idx.shape == (2, slvr.nsrc, slvr.ntime, slvr.nbl, slvr.nchan, 4)

            result = self.jones_multiply_hermitian_transpose(
                ekb_sqrt_idx[0], ekb_sqrt_idx[1],
                slvr.nsrc*slvr.ntime*slvr.nbl*slvr.nchan) \
                    .reshape(slvr.nsrc, slvr.ntime, slvr.nbl, slvr.nchan, 4)

            # Multiply in Gaussian Shape Terms
            if slvr.ngsrc > 0:
                src_beg = slvr.npsrc
                src_end = slvr.npsrc + slvr.ngsrc
                gauss_shape = self.compute_gaussian_shape()
                result[src_beg:src_end,:,:,:,:] *= gauss_shape[:,:,:,:,np.newaxis]

            # Multiply in Sersic Shape Terms
            if slvr.nssrc > 0:
                src_beg = slvr.npsrc + slvr.ngsrc
                src_end = slvr.npsrc + slvr.ngsrc + slvr.nssrc
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

        slvr = self.solver

        if ekb_jones is None:
            ekb_jones = self.compute_ekb_jones_per_bl()

        want_shape = (slvr.nsrc, slvr.ntime, slvr.nbl, slvr.nchan, 4)
        assert ekb_jones.shape == want_shape, \
            'Expected shape %s. Got %s instead.' % \
            (want_shape, ekb_jones.shape)

        if slvr.nsrc == 1:
            # Due to this bug
            # https://github.com/pydata/numexpr/issues/79
            # numexpr may not reduce a source axis of size 1
            # Work around this
            vis = ekb_jones.squeeze(0)
        else:
            vis = ne.evaluate('sum(ebk,0)',
                {'ebk': ekb_jones }) \
                .astype(slvr.ct)

        assert vis.shape == (slvr.ntime, slvr.nbl, slvr.nchan, 4)

        return vis

    def compute_gekb_vis(self, ekb_vis=None):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (ntime,nbl,nchan,4) matrix of complex scalars.
        """

        slvr = self.solver

        if ekb_vis is None:
            ekb_vis = self.compute_ekb_vis()

        want_shape = (slvr.ntime, slvr.nbl, slvr.nchan, 4)
        assert ekb_vis.shape == want_shape, \
            'Expected shape %s. Got %s instead.' % \
            (want_shape, ekb_vis.shape)

        ap = slvr.get_ap_idx(chan=True)
        g_term = slvr.G_term_cpu[ap]

        assert g_term.shape == (2, slvr.ntime, slvr.nbl, slvr.nchan, 4)

        result = self.jones_multiply_hermitian_transpose(
            g_term[0], ekb_vis,
            slvr.ntime*slvr.nbl*slvr.nchan) \
                .reshape(slvr.ntime, slvr.nbl, slvr.nchan, 4)

        result = self.jones_multiply_hermitian_transpose(
            result, g_term[1],
            slvr.ntime*slvr.nbl*slvr.nchan) \
                .reshape(slvr.ntime, slvr.nbl, slvr.nchan, 4)

        return result

    def compute_chi_sqrd_sum_terms(self, vis=None, bayes_data=None, weight_vector=False):
        """
        Computes the terms of the chi squared sum,
        but does not perform the sum itself.

        Parameters:
            weight_vector : boolean
                True if the chi squared test terms
                should be computed with a noise vector

        Returns a (ntime,nbl,nchan) matrix of floating point scalars.
        """
        slvr = self.solver

        try:
            if vis is None: vis = self.compute_ekb_vis()
            if bayes_data is None: bayes_data = slvr.bayes_data_cpu

            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = ne.evaluate('vis - bayes', {
                'vis': vis,
                'bayes': bayes_data })
            assert d.shape == (slvr.ntime, slvr.nbl, slvr.nchan, 4)

            # Square of the real and imaginary components
            re = ne.evaluate('re**2', {'re': d.real})
            im = ne.evaluate('im**2', {'im': d.imag})
            wv = slvr.weight_vector_cpu

            # Multiply by the weight vector if required
            if weight_vector is True:
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
            assert chi_sqrd_terms.shape == (slvr.ntime, slvr.nbl, slvr.nchan)

            return chi_sqrd_terms

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_chi_sqrd(self, vis=None, bayes_data=None, weight_vector=False):
        """ Computes the chi squared value.

        Parameters:
            weight_vector : boolean
                True if the chi squared test
                should be computed with a noise vector

        Returns a floating point scalar values
        """
        slvr = self.solver

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum
        try:
            if vis is None: vis = self.compute_ekb_vis()
            if bayes_data is None: bayes_data = slvr.bayes_data_cpu

            chi_sqrd_terms = self.compute_chi_sqrd_sum_terms(
                vis=vis, bayes_data=bayes_data, weight_vector=weight_vector)
            term_sum = ne.evaluate('sum(terms)', {'terms': chi_sqrd_terms})
            return term_sum if weight_vector is True \
                else term_sum / slvr.sigma_sqrd
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_biro_chi_sqrd(self, vis=None, bayes_data=None, weight_vector=False):
        slvr = self.solver

        if vis is None: vis = self.compute_ekb_vis()
        if bayes_data is None: bayes_data = slvr.bayes_data_cpu

        return self.compute_chi_sqrd(vis=vis, bayes_data=bayes_data,
            weight_vector=weight_vector)
