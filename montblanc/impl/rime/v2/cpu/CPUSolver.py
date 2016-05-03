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

import montblanc.util as mbu

import numexpr as ne
import numpy as np

from montblanc.solvers import MontblancNumpySolver

class CPUSolver(MontblancNumpySolver):
    def __init__(self, slvr_cfg):
        super(CPUSolver, self).__init__(slvr_cfg)

        # Monkey patch these functions onto the object
        # TODO: Remove this when deprecating v2.
        from montblanc.impl.rime.v2.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        from montblanc.impl.rime.v2.config import (A, P)

        self.register_default_dimensions()
        self.register_properties(P)
        self.register_arrays(A)

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (ntime, nbl, ngsrc, nchan) matrix of floating point scalars.
        """
        ntime, nbl, ngsrc = self.dim_local_size('ntime', 'nbl', 'ngsrc')

        ap = self.ap_idx()

        # Calculate per baseline u from per antenna u
        u = self.uvw[0][ap]
        u = ne.evaluate('aq-ap', {'ap': u[0], 'aq': u[1]})

        # Calculate per baseline v from per antenna v
        v = self.uvw[1][ap]
        v = ne.evaluate('aq-ap', {'ap': v[0], 'aq': v[1]})

        # Calculate per baseline w from per antenna w
        w = self.uvw[2][ap]
        w = ne.evaluate('aq-ap', {'ap': w[0], 'aq': w[1]})

        el = self.gauss_shape[0]
        em = self.gauss_shape[1]
        R = self.gauss_shape[2]

        # OK, try obtain the same results with the fwhm factored out!
        # u1 = u*em - v*el
        # v1 = u*el + v*em
        u1 = ne.evaluate('u_em - v_el',
            {'u_em': np.outer(u, em), 'v_el': np.outer(v, el)})\
            .reshape(ntime,nbl,ngsrc)
        v1 = ne.evaluate('u_el + v_em', {
            'u_el' : np.outer(u,el), 'v_em' : np.outer(v, em)})\
            .reshape(ntime,nbl,ngsrc)

        # Obvious given the above reshape
        assert u1.shape == (ntime, nbl, ngsrc)
        assert v1.shape == (ntime, nbl, ngsrc)

        return ne.evaluate('exp(-((u1*scale_uv*R)**2 + (v1*scale_uv)**2))',
            local_dict={
                'u1':u1[:,:,:,np.newaxis],
                'v1':v1[:,:,:,np.newaxis],
                'scale_uv':(self.gauss_scale/
                    self.wavelength)[np.newaxis,np.newaxis,np.newaxis,:],
                'R':R[np.newaxis,np.newaxis,:,np.newaxis]})

    def compute_sersic_shape(self):
        """
        Compute the shape values for the sersic (exponential) sources.

        Returns a (ntime, nbl, nssrc, nchan) matrix of floating point scalars.
        """        
        ntime, nbl, nchan, nssrc = self.dim_local_size('ntime', 'nbl', 'nchan', 'nssrc')

        ap = self.ap_idx()

        # Calculate per baseline u from per antenna u
        u = self.uvw[0][ap]
        u = ne.evaluate('aq-ap', {'ap': u[0], 'aq': u[1]})

        # Calculate per baseline v from per antenna v
        v = self.uvw[1][ap]
        v = ne.evaluate('aq-ap', {'ap': v[0], 'aq': v[1]})

        # Calculate per baseline w from per antenna w
        w = self.uvw[2][ap]
        w = ne.evaluate('aq-ap', {'ap': w[0], 'aq': w[1]})

        e1 = self.sersic_shape[0]
        e2 = self.sersic_shape[1]
        R = self.sersic_shape[2]

        # OK, try obtain the same results with the fwhm factored out!
        # u1 = u*(1+e1) - v*e2
        # v1 = u*e2 + v*(1-e1)
        u1 = ne.evaluate('u_1_e1 + v_e2',
            {'u_1_e1': np.outer(u,np.ones(nssrc)+e1), 'v_e2' : np.outer(v, e2)})\
            .reshape(ntime,nbl,nssrc)
        v1 = ne.evaluate('u_e2 + v_1_e1', {
            'u_e2' : np.outer(u,e2), 'v_1_e1' : np.outer(v,np.ones(nssrc)-e1)})\
            .reshape(ntime,nbl,nssrc)

        # Obvious given the above reshape
        assert u1.shape == (ntime, nbl, nssrc)
        assert v1.shape == (ntime, nbl, nssrc)

        den = ne.evaluate('1 + (u1*scale_uv*R)**2 + (v1*scale_uv*R)**2',
            local_dict={
                'u1': u1[:, :, :, np.newaxis],
                'v1': v1[:, :, :, np.newaxis],
                'scale_uv': (self.two_pi / self.wavelength)
                    [np.newaxis, np.newaxis, np.newaxis, :],
                'R': (R / (1 - e1 * e1 - e2 * e2))
                    [np.newaxis,np.newaxis,:,np.newaxis]})\
                .reshape(ntime, nbl, nssrc, nchan)

        return ne.evaluate('1/(den*sqrt(den))',
            { 'den' : den[:, :, :, :] })

    def compute_k_jones_scalar_per_ant(self):
        """
        Computes the scalar K (phase) term of the RIME per antenna.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """        
        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')

        wave = self.wavelength

        u, v, w = self.uvw[0], self.uvw[1], self.uvw[2]
        l, m = self.lm[0], self.lm[1]
        alpha = self.brightness[4]

        # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
        n = ne.evaluate('sqrt(1. - l**2 - m**2) - 1.',
            {'l': l, 'm': m})

        # w*n+v*m+u*l. Outer product creates array of dim ntime x na x nsrcs
        phase = (np.outer(w, n) + np.outer(v, m) + np.outer(u, l)) \
                .reshape(ntime, na, nsrc)
        assert phase.shape == (ntime, na, nsrc)

        # e^(2*pi*sqrt(u*l+v*m+w*n)/wavelength).
        # Dim. na x ntime x nchan x nsrcs
        phase = ne.evaluate('exp(-2*pi*1j*p/wl)', {
            'p': phase[:, :, :, np.newaxis],
            'wl': wave[np.newaxis, np.newaxis, np.newaxis, :],
            'pi': np.pi
        })

        assert phase.shape == (ntime, na, nsrc, nchan)

        # Dimension ntime x nsrc x nchan. Use 0.5*alpha here so that
        # when the other antenna term is multiplied with this one, we
        # end up with the full power term. sqrt(n)*sqrt(n) == n.
        power = ne.evaluate('(rw/wl)**(0.5*a)', {
            'rw': self.ref_wave,
            'wl': wave[np.newaxis, np.newaxis, :],
            'a': alpha[:, :, np.newaxis]
        })
        assert power.shape == (ntime, nsrc, nchan)

        return ne.evaluate('phs*p', {
            'phs': phase,
            'p': power[:, np.newaxis, :, :]
        }).astype(self.ct)  # Need a cast since numexpr upcasts

    def compute_k_jones_scalar_per_bl(self):
        """
        Computes the scalar K (phase) term of the RIME per baseline.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """        
        npsrc, ngsrc, nssrc = self.dim_local_size('npsrc', 'ngsrc', 'nssrc')

        # Re-arrange per antenna terms into per baseline antenna pair values
        ap = self.ap_idx(src=True, chan=True)
        k_jones = self.compute_k_jones_scalar_per_ant()[ap]

        k_jones_per_bl = k_jones[0]*k_jones[1].conj()

        # Add in the shape terms of the gaussian sources.
        if ngsrc > 0:
            src_beg = npsrc
            src_end = npsrc + ngsrc
            k_jones_per_bl[:, :, src_beg:src_end, :] *= \
                self.compute_gaussian_shape()
            # TODO: Would like to do this, but fails because of
            # https://github.com/pydata/numexpr/issues/155
            #gsrc_view = k_jones_per_bl[:,:,npsrc:,:]
            #gshape = self.compute_gaussian_shape()
            #ne.evaluate('kjones*complex(gshape.real,0.0)',
            #    {'kjones' : gsrc_view, 'gshape':gshape }, out=gsrc_view)

        # Add in the shape terms of the sersic sources.
        if nssrc > 0:
            src_beg = npsrc+ngsrc
            src_end = npsrc+ngsrc+nssrc
            k_jones_per_bl[:, :, src_beg:src_end:, :] *= \
                self.compute_sersic_shape()

        return k_jones_per_bl

    def compute_e_jones_scalar_per_ant(self):
        """
        Computes the scalar E (analytic cos^3) term per antenna.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """        
        nsrc, ntime, na, nchan = self.dim_local_size('nsrc', 'ntime', 'na', 'nchan')

        # Compute the offsets for different antenna
        # Broadcasting here produces, ntime x na x  nsrc
        E_p = ne.evaluate('sqrt((l - lp)**2 + (m - mp)**2)', {
            'l': self.lm[0], 'm': self.lm[1],
            'lp': self.point_errors[0, :, :, np.newaxis],
            'mp': self.point_errors[1, :, :, np.newaxis]
        })

        assert E_p.shape == (ntime, na, nsrc)

        # Broadcasting here produces, ntime x nbl x nsrc x nchan
        E_p = ne.evaluate('E*bw*1e-9*wl', {
            'E': E_p[:, :, :, np.newaxis], 'bw': self.beam_width,
            'wl': self.wavelength[np.newaxis, np.newaxis, np.newaxis, :]
        })

        # Clip the beam
        np.clip(E_p, np.finfo(self.ft).min, self.beam_clip, E_p)
        # Cosine it, cube it and cast because of numexpr
        E_p = ne.evaluate('cos(E)**3', {'E': E_p}).astype(self.ct)

        assert E_p.shape == (ntime, na, nsrc, nchan)

        return E_p

    def compute_e_jones_scalar_per_bl(self):
        """
        Computes the scalar E (analytic cos^3) term per baseline.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """        

        # Re-arrange per antenna terms into per baseline antenna pair values
        ap = self.ap_idx(src=True, chan=True)
        e_jones = self.compute_e_jones_scalar_per_ant()[ap]

        return e_jones[0]*e_jones[1].conj()

    def compute_ek_jones_scalar_per_ant(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """
        return (self.compute_k_jones_scalar_per_ant() *
            self.compute_e_jones_scalar_per_ant())

    def compute_ek_jones_scalar_per_bl(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        per_bl_ek_scalar = (self.compute_k_jones_scalar_per_bl() *
            self.compute_e_jones_scalar_per_bl())

        return per_bl_ek_scalar

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,ntime,nsrc) matrix of complex scalars.
        """        
        nsrc, ntime = self.dim_local_size('nsrc', 'ntime')

        # Create the brightness matrix. Dim 4 x ntime x nsrcs
        B = self.ct([
            # fI+fQ + 0j
            self.brightness[0]+self.brightness[1] + 0j,
            # fU + fV*1j
            self.brightness[2] + 1j*self.brightness[3],
            # fU - fV*1j
            self.brightness[2] - 1j*self.brightness[3],
            # fI-fQ + 0j
            self.brightness[0]-self.brightness[1] + 0j])
        assert B.shape == (4, ntime, nsrc)

        return B

    def compute_ebk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """        
        nsrc, ntime, nbl, nchan = self.dim_local_size('nsrc', 'ntime', 'nbl', 'nchan')

        per_bl_ek_scalar = self.compute_ek_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_ek_scalar[np.newaxis, :, :, :, :] * \
            b_jones[:, :, np.newaxis, :, np.newaxis]
        assert jones.shape == (4, ntime, nbl, nsrc, nchan)

        return jones

    def compute_bk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """        
        nsrc, ntime, nbl, nchan = self.dim_local_size('nsrc', 'ntime', 'nbl', 'nchan')

        per_bl_k_scalar = self.compute_k_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_k_scalar[np.newaxis, :, :, :, :] * \
            b_jones[:, :, np.newaxis, :, np.newaxis]
        assert jones.shape == (4, ntime, nbl, nsrc, nchan)

        return jones

    def compute_ebk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """        
        nsrc, ntime, nbl, nchan = self.dim_local_size('nsrc', 'ntime', 'nbl', 'nchan')

        ebk_jones = self.compute_ebk_jones()

        if nsrc == 1:
            # Due to this bug
            # https://github.com/pydata/numexpr/issues/79
            # numexpr may not reduce a source axis of size 1
            # Work around this
            vis = ebk_jones.squeeze(3)
        else:
            vis = ne.evaluate('sum(ebk,3)',
                {'ebk': ebk_jones }) \
                .astype(self.ct)

        assert vis.shape == (4, ntime, nbl, nchan)

        # Zero any flagged visibilities
        vis[self.flag > 0] = 0

        return vis

    def compute_bk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar K term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """
        
        ntime, nbl, nchan = self.dim_local_size(ntime, nbl, nchan)

        vis = ne.evaluate('sum(bk,3)', {'bk': self.compute_bk_jones()})\
            .astype(self.ct)
        assert vis.shape == (4, ntime, nbl, nchan)

        return vis

    def compute_chi_sqrd_sum_terms(self, vis=None):
        """
        Computes the terms of the chi squared sum,
        but does not itself perform the sum.

        Returns a (ntime,nbl,nchan) matrix of floating point scalars.
        """        
        ntime, nbl, nchan = self.dim_local_size('ntime', 'nbl', 'nchan')

        if vis is None: vis = self.compute_ebk_vis()

        # Take the difference between the observed and model visibilities
        # (4,nbl,nchan,ntime)
        d = ne.evaluate('mvis - (ovis*where(flag > 0, 0, 1))', {
            'mvis': vis,
            'ovis': self.observed_vis,
            'flag': self.flag})
        assert d.shape == (4, ntime, nbl, nchan)

        # Square of the real and imaginary components
        re = ne.evaluate('re**2', {'re': d.real})
        im = ne.evaluate('im**2', {'im': d.imag})

        # Multiply by the weight vector if required
        if self.use_weight_vector() is True:
            wv = self.weight_vector
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
        assert chi_sqrd_terms.shape == (ntime, nbl, nchan)

        return chi_sqrd_terms

    def compute_chi_sqrd(self, chi_sqrd_terms=None):
        """ Computes the floating point scalar chi squared value. """

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum
        if chi_sqrd_terms is None:
            chi_sqrd_terms = self.compute_chi_sqrd_sum_terms()

        term_sum = ne.evaluate('sum(terms)', {'terms': chi_sqrd_terms})
        return (term_sum if self.use_weight_vector() 
            else term_sum / self.sigma_sqrd)

    def solve(self):
        """ Solve the RIME """

        self.jones_scalar[:] = self.compute_ek_jones_scalar_per_ant()
        
        self.model_vis[:] = self.compute_ebk_vis()
        
        self.chi_sqrd_result[:] = self.compute_chi_sqrd_sum_terms(
            self.model_vis)
        
        self.set_X2(self.compute_chi_sqrd(
            self.chi_sqrd_result))