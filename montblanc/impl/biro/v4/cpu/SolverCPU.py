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
            u = slvr.uvw_cpu[0][ap]
            u = ne.evaluate('ap-aq', {'ap': u[1], 'aq': u[0]})

            # Calculate per baseline v from per antenna v
            v = slvr.uvw_cpu[1][ap]
            v = ne.evaluate('ap-aq', {'ap': v[1], 'aq': v[0]})

            # Calculate per baseline w from per antenna w
            w = slvr.uvw_cpu[2][ap]
            w = ne.evaluate('ap-aq', {'ap': w[1], 'aq': w[0]})

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

            return ne.evaluate('exp(-((u1*scale_uv*R)**2 + (v1*scale_uv)**2))',
                local_dict={
                    'u1':u1[:,:,:,np.newaxis],
                    'v1':v1[:,:,:,np.newaxis],
                    'scale_uv':(slvr.gauss_scale/slvr.wavelength_cpu)[np.newaxis,np.newaxis,np.newaxis,:],
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
            u = slvr.uvw_cpu[0][ap]
            u = ne.evaluate('ap-aq', {'ap': u[1], 'aq': u[0]})

            # Calculate per baseline v from per antenna v
            v = slvr.uvw_cpu[1][ap]
            v = ne.evaluate('ap-aq', {'ap': v[1], 'aq': v[0]})

            # Calculate per baseline w from per antenna w
            w = slvr.uvw_cpu[2][ap]
            w = ne.evaluate('ap-aq', {'ap': w[1], 'aq': w[0]})

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

            den = ne.evaluate('1 + (u1*scale_uv*R)**2 + (v1*scale_uv*R)**2',
                local_dict={
                    'u1': u1[:, :, :, np.newaxis],
                    'v1': v1[:, :, :, np.newaxis],
                    'scale_uv': (slvr.two_pi / slvr.wavelength_cpu)
                        [np.newaxis, np.newaxis, np.newaxis, :],
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
            wave = slvr.wavelength_cpu

            u, v, w = slvr.uvw_cpu[0], slvr.uvw_cpu[1], slvr.uvw_cpu[2]
            l, m = slvr.lm_cpu[0], slvr.lm_cpu[1]

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
            n = ne.evaluate('sqrt(1. - l**2 - m**2) - 1.',
                {'l': l, 'm': m})

            # w*n+v*m+u*l. Outer product creates array of dim nsrcs x ntime x na
            phase = (np.outer(n, w) + np.outer(m, v) + np.outer(l, u)) \
                    .reshape(slvr.nsrc, slvr.ntime, slvr.na)

            # e^(2*pi*sqrt(u*l+v*m+w*n)/wavelength).
            # Dim. ntime x na x nchan x nsrcs
            cplx_phase = ne.evaluate('exp(2*pi*1j*p/wl)', {
                'p': phase[:, :, :, np.newaxis],
                'wl': wave[np.newaxis, np.newaxis, np.newaxis, :],
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
        assert b_jones.shape == (4, slvr.nsrc, slvr.ntime)

        result = k_jones[np.newaxis,:,:,:,:]*b_jones[:,:,:,np.newaxis,np.newaxis]
        assert result.shape == (4, slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

        return result

    def compute_k_jones_scalar_per_bl(self):
        """
        Computes the scalar K (phase) term of the RIME per baseline.

        Returns a (nsrc,ntime,nbl,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Re-arrange per antenna terms into per baseline antenna pair values
            ap = slvr.get_ap_idx(src=True, chan=True)
            k_jones = self.compute_k_jones_scalar_per_ant()[ap]

            k_jones_per_bl = k_jones[1]*k_jones[0].conj()

            # Add in the shape terms of the gaussian sources.
            if slvr.ngsrc > 0:
                src_beg = slvr.npsrc
                src_end = slvr.npsrc + slvr.ngsrc
                k_jones_per_bl[src_beg:src_end, :, :  :] *=\
                    self.compute_gaussian_shape()
                # TODO: Would like to do this, but fails because of
                # https://github.com/pydata/numexpr/issues/155
                #gsrc_view = k_jones_per_bl[:,:,slvr.npsrc:,:]
                #gshape = self.compute_gaussian_shape()
                #ne.evaluate('kjones*complex(gshape.real,0.0)',
                #    {'kjones' : gsrc_view, 'gshape':gshape }, out=gsrc_view)

            # Add in the shape terms of the sersic sources.
            if slvr.nssrc > 0:
                src_beg = slvr.npsrc+slvr.ngsrc
                src_end = slvr.npsrc+slvr.ngsrc+slvr.nssrc
                k_jones_per_bl[src_beg:src_end:, :, :,  :] *=  \
                    self.compute_sersic_shape()

            return k_jones_per_bl
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_e_jones_scalar_per_ant(self):
        """
        Computes the scalar E (analytic cos^3) term per antenna.

        Returns a (nsrc,ntime,na,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Compute the offsets for different antenna
            # Broadcasting here produces, nsrc x ntime x na
            E_p = ne.evaluate('sqrt((l - lp)**2 + (m - mp)**2)', {
                'l': slvr.lm_cpu[0, :, np.newaxis, np.newaxis],
                'm': slvr.lm_cpu[1, :, np.newaxis, np.newaxis],
                'lp': slvr.point_errors_cpu[0, np.newaxis, :, :],
                'mp': slvr.point_errors_cpu[1, np.newaxis, :, :]
            })

            assert E_p.shape == (slvr.nsrc, slvr.ntime, slvr.na)

            # Broadcasting here produces, nsrc x ntime x nbl x nchan
            E_p = ne.evaluate('E*bw*1e-9*wl', {
                'E': E_p[:, :, :, np.newaxis], 'bw': slvr.beam_width,
                'wl': slvr.wavelength_cpu[np.newaxis, np.newaxis, np.newaxis, :]
            })

            assert E_p.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

            # Clip the beam
            np.clip(E_p, np.finfo(slvr.ft).min, slvr.beam_clip, E_p)
            # Cosine it, cube it and cast because of numexpr
            E_p = ne.evaluate('cos(E)**3', {'E': E_p}).astype(slvr.ct)

            assert E_p.shape == (slvr.nsrc, slvr.ntime, slvr.na, slvr.nchan)

            return E_p
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_e_jones_scalar_per_bl(self):
        """
        Computes the scalar E (analytic cos^3) term per baseline.

        Returns a (nsrc,ntime,nbl,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Re-arrange per antenna terms into per baseline antenna pair values
            ap = slvr.get_ap_idx(src=True, chan=True)
            e_jones = self.compute_e_jones_scalar_per_ant()[ap]

            return e_jones[1]*e_jones[0].conj()
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_ek_jones_scalar_per_ant(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (nsrc,ntime,na,nchan) matrix of complex scalars.
        """
        return self.compute_k_jones_scalar_per_ant() * \
            self.compute_e_jones_scalar_per_ant()

    def compute_ek_jones_scalar_per_bl(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (nsrc,ntime,nbl,nchan) matrix of complex scalars.
        """
        per_bl_ek_scalar = self.compute_k_jones_scalar_per_bl() * \
            self.compute_e_jones_scalar_per_bl()

        return per_bl_ek_scalar

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,nsrc,ntime) matrix of complex scalars.
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

            return B

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_b_sqrt_jones(self):
        """
        Computes the B sqrt term of the RIME.

        Returns a (4,nsrc,ntime) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            B = self.compute_b_jones()
            I = slvr.stokes_cpu[:,:,0]
            Q = slvr.stokes_cpu[:,:,1]
            U = slvr.stokes_cpu[:,:,2]
            V = slvr.stokes_cpu[:,:,3]

            trace = 2*I
            det = I**2 - Q**2 - U**2 - V**2

            # We don't have a solution for some matrices
            # But if both trace and determinant are zero
            # the matrix itself must be zero.
            mask = np.logical_and(trace == 0, det == 0)

            s = np.sqrt(trace.astype(slvr.ct))
            t = np.sqrt(det.astype(slvr.ct) + 2*s)

            # Avoid infs and nans from divide by zero
            t[mask] = 1

            B_sqrt = np.array([B[0] + s,
                B[1],
                B[2],
                B[3] + s]) / t

            assert B_sqrt.shape == (4, slvr.nsrc, slvr.ntime)

            return B_sqrt

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_ebk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,nsrc,ntime,nbl,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        per_bl_ek_scalar = self.compute_ek_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_ek_scalar[np.newaxis, :, :, :, :] * \
            b_jones[:, :, :, np.newaxis, np.newaxis]
        assert jones.shape == (4, slvr.nsrc, slvr.ntime, slvr.nbl, slvr.nchan)

        return jones

    def compute_bk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,nsrc,ntime,nbl,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        per_bl_k_scalar = self.compute_k_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_k_scalar[np.newaxis, :, :, :, :] * \
            b_jones[:, :, :, np.newaxis, np.newaxis]
        assert jones.shape == (4, slvr.nsrc, slvr.ntime, slvr.nbl, slvr.nchan)

        return jones

    def compute_ebk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        vis = ne.evaluate('sum(ebk,1)', {'ebk': self.compute_ebk_jones()})\
            .astype(slvr.ct)
        assert vis.shape == (4, slvr.ntime, slvr.nbl, slvr.nchan)

        return vis

    def compute_bk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar K term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        vis = ne.evaluate('sum(bk,3)', {'bk': self.compute_bk_jones()})\
            .astype(slvr.ct)
        assert vis.shape == (4, slvr.ntime, slvr.nbl, slvr.nchan)

        return vis

    def compute_chi_sqrd_sum_terms(self, weight_vector=False):
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
            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = ne.evaluate('vis - bayes', {
                'vis': slvr.vis_cpu,
                'bayes': slvr.bayes_data_cpu})
            assert d.shape == (4, slvr.ntime, slvr.nbl, slvr.nchan)

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
            re_sum = ne.evaluate('sum(re,0)', {'re': re})
            im_sum = ne.evaluate('sum(im,0)', {'im': im})
            chi_sqrd_terms = ne.evaluate('re_sum + im_sum',
                {'re_sum': re_sum, 'im_sum': im_sum})
            assert chi_sqrd_terms.shape == (slvr.ntime, slvr.nbl, slvr.nchan)

            return chi_sqrd_terms

        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_chi_sqrd(self, weight_vector=False):
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
            chi_sqrd_terms = self.compute_chi_sqrd_sum_terms(
                weight_vector=weight_vector)
            term_sum = ne.evaluate('sum(terms)', {'terms': chi_sqrd_terms})
            return term_sum if weight_vector is True \
                else term_sum / slvr.sigma_sqrd
        except AttributeError as e:
            mbu.rethrow_attribute_exception(e)

    def compute_biro_chi_sqrd(self, weight_vector=False):
        slvr = self.solver
        slvr.vis_cpu = self.compute_ebk_vis()
        return self.compute_chi_sqrd(weight_vector=weight_vector)
