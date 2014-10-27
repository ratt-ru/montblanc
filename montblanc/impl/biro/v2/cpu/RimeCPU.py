import numpy as np

def rethrow_attribute_exception(e):
    raise AttributeError, '%s. The appropriate numpy array has not ' \
        'been set on the shared data object. You need to set ' \
        'store_cpu=True on your shared data object ' \
        'as well as call the transfer_* method for this to work.' % e

class RimeCPU(object):
    def __init__(self, solver):
        self.solver = solver

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (ntime, nbl, ngsrc, nchan) matrix of floating point scalars.
        """

        slvr = self.solver

        try:
            ant0, ant1 = slvr.get_flat_ap_idx()

            uvw = slvr.uvw_cpu.reshape(3,slvr.ntime*slvr.na)
            u = (uvw[0][ant1] - uvw[0][ant0]).reshape(slvr.ntime, slvr.nbl)
            v = (uvw[1][ant1] - uvw[1][ant0]).reshape(slvr.ntime, slvr.nbl)
            w = (uvw[2][ant1] - uvw[2][ant0]).reshape(slvr.ntime, slvr.nbl)

            el = slvr.gauss_shape_cpu[0]
            em = slvr.gauss_shape_cpu[1]
            R = slvr.gauss_shape_cpu[2]

            # OK, try obtain the same results with the fwhm factored out!
            # u1 = u*em - v*el
            # v1 = u*el + v*em
            u1 = (np.outer(u, em) - np.outer(v, el)) \
                .reshape(slvr.ntime,slvr.nbl,slvr.ngsrc)
            v1 = (np.outer(u, el) + np.outer(v, em)) \
                .reshape(slvr.ntime,slvr.nbl,slvr.ngsrc)

            # Obvious given the above reshape
            assert u1.shape == (slvr.ntime, slvr.nbl, slvr.ngsrc)
            assert v1.shape == (slvr.ntime, slvr.nbl, slvr.ngsrc)

            # Construct the scaling factor, this includes the wavelength/frequency
            # into the mix.
            scale_uv = slvr.gauss_scale/slvr.wavelength_cpu
            assert scale_uv.shape == (slvr.nchan,)

            # Multiply u1 and v1 by the scaling factor
            u1 = u1[:,:,:,np.newaxis]*scale_uv[np.newaxis,np.newaxis,np.newaxis,:]
            v1 = v1[:,:,:,np.newaxis]*scale_uv[np.newaxis,np.newaxis,np.newaxis,:]
            # u1 *= R, the ratio of the gaussian axis
            u1 *= R[np.newaxis,np.newaxis,:,np.newaxis]

            return np.exp(-(u1**2 + v1**2))

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_k_jones_scalar_per_ant(self):
        """
        Computes the scalar K (phase) term of the RIME per antenna.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            wave = slvr.wavelength_cpu

            u, v, w = slvr.uvw_cpu[0], slvr.uvw_cpu[1], slvr.uvw_cpu[2]
            l, m = slvr.lm_cpu[0], slvr.lm_cpu[1]
            alpha = slvr.brightness_cpu[4]

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
            n = np.sqrt(1. - l**2 - m**2) - 1.

            # w*n+v*m+u*l. Outer product creates array of dim ntime x na x nsrcs
            phase = (np.outer(w,n) + np.outer(v, m) + np.outer(u, l)) \
                    .reshape(slvr.ntime, slvr.na, slvr.nsrc)
            assert phase.shape == (slvr.ntime, slvr.na, slvr.nsrc)            

            # e^(2*pi*sqrt(u*l+v*m+w*n)/wavelength). Dim. na x ntime x nchan x nsrcs 
            phase = np.exp((2*np.pi*1j*phase)[:,:,:,np.newaxis]/ \
                wave[np.newaxis,np.newaxis,np.newaxis,:])
            assert phase.shape == (slvr.ntime, slvr.na, slvr.nsrc, slvr.nchan)

            # Dimension ntime x nsrc x nchan. Use 0.5*alpha here so that
            # when the other antenna term is multiplied with this one, we
            # end up with the full power term. sqrt(n)*sqrt(n) == n.
            power = np.power(slvr.ref_wave/wave[np.newaxis,np.newaxis,:],
                0.5*alpha[:,:,np.newaxis])
            assert power.shape == (slvr.ntime, slvr.nsrc, slvr.nchan)

            phase *= power[:,np.newaxis,:,:]
            assert phase.shape == (slvr.ntime, slvr.na, slvr.nsrc, slvr.nchan) 

            return phase

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_k_jones_scalar_per_bl(self):
        """
        Computes the scalar K (phase) term of the RIME per baseline.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            ant0, ant1 = slvr.get_flat_ap_idx(src=True,chan=True)
            k_jones = self.compute_k_jones_scalar_per_ant().flatten()

            k_jones_per_bl = (k_jones[ant1]*k_jones[ant0].conj())\
                .reshape(slvr.ntime,slvr.nbl,slvr.nsrc,slvr.nchan)

            # Add in the shape terms of the gaussian sources.
            if slvr.ngsrc > 0:
                k_jones_per_bl[:,:,slvr.npsrc:,:] *= self.compute_gaussian_shape()

            return k_jones_per_bl
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_e_jones_scalar_per_ant(self):
        """
        Computes the scalar E (analytic cos^3) term per antenna.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Compute the offsets for different antenna
            # Broadcasting here produces, ntime x na x  nsrc
            l_diff = slvr.lm_cpu[0] - slvr.point_errors_cpu[0,:,:,np.newaxis]
            m_diff = slvr.lm_cpu[1] - slvr.point_errors_cpu[1,:,:,np.newaxis]
            E_p = np.sqrt(l_diff**2 + m_diff**2)

            assert E_p.shape == (slvr.ntime, slvr.na, slvr.nsrc)

            # Broadcasting here produces, ntime x nbl x nsrc x nchan
            E_p = E_p[:,:,:,np.newaxis]*slvr.beam_width*1e-9*\
                slvr.wavelength_cpu[np.newaxis,np.newaxis,np.newaxis,:]
            np.clip(E_p, np.finfo(slvr.ft).min, slvr.beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (slvr.ntime, slvr.na, slvr.nsrc, slvr.nchan)

            return E_p
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_e_jones_scalar_per_bl(self):
        """
        Computes the scalar E (analytic cos^3) term per baseline.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            ant0, ant1 = slvr.get_flat_ap_idx(src=True,chan=True)
            e_jones = self.compute_e_jones_scalar_per_ant().flatten()

            return (e_jones[ant1]*e_jones[ant0].conj())\
                .reshape(slvr.ntime,slvr.nbl,slvr.nsrc,slvr.nchan)
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_ek_jones_scalar_per_ant(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (ntime,na,nsrc,nchan) matrix of complex scalars.
        """
        return self.compute_k_jones_scalar_per_ant()*\
            self.compute_e_jones_scalar_per_ant()

    def compute_ek_jones_scalar_per_bl(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Returns a (ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver

        per_bl_ek_scalar = self.compute_k_jones_scalar_per_bl() * \
            self.compute_e_jones_scalar_per_bl()

        return per_bl_ek_scalar

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Create the brightness matrix. Dim 4 x ntime x nsrcs
            B = slvr.ct([
                slvr.brightness_cpu[0]+slvr.brightness_cpu[1] + 0j,     # fI+fQ + 0j
                slvr.brightness_cpu[2] + 1j*slvr.brightness_cpu[3],     # fU + fV*1j
                slvr.brightness_cpu[2] - 1j*slvr.brightness_cpu[3],     # fU - fV*1j
                slvr.brightness_cpu[0]-slvr.brightness_cpu[1] + 0j])    # fI-fQ + 0j
            assert B.shape == (4, slvr.ntime, slvr.nsrc)

            return B

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_ebk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver
        
        per_bl_ek_scalar = self.compute_ek_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_ek_scalar[np.newaxis,:,:,:,:]*\
            b_jones[:,:,np.newaxis,:,np.newaxis]
        assert jones.shape == (4,slvr.ntime,slvr.nbl,slvr.nsrc,slvr.nchan)

        return jones

    def compute_bk_jones(self):
        """
        Computes the jones matrices based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nsrc,nchan) matrix of complex scalars.
        """
        slvr = self.solver
        
        per_bl_k_scalar = self.compute_k_jones_scalar_per_bl()
        b_jones = self.compute_b_jones()

        jones = per_bl_k_scalar[np.newaxis,:,:,:,:]*\
            b_jones[:,:,np.newaxis,:,np.newaxis]
        assert jones.shape == (4,slvr.ntime,slvr.nbl,slvr.nsrc,slvr.nchan)

        return jones

    def compute_ebk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        vis = np.add.reduce(self.compute_ebk_jones(),axis=3)
        assert vis.shape == (4,slvr.ntime,slvr.nbl,slvr.nchan)

        return vis

    def compute_bk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar K term and the 2x2 B term.

        Returns a (4,ntime,nbl,nchan) matrix of complex scalars.
        """

        slvr = self.solver

        vis = np.add.reduce(self.compute_bk_jones(),axis=3)
        assert vis.shape == (4,slvr.ntime,slvr.nbl,slvr.nchan)

        return vis

    def compute_chi_sqrd_sum_terms(self, weight_vector=False):
        """
        Computes the terms of the chi squared sum, but does not perform the sum itself.

        Parameters:
            weight_vector : boolean
                True if the chi squared test terms should be computed with a noise vector

        Returns a (ntime,nbl,nchan) matrix of floating point scalars.
        """
        slvr = self.solver

        try:
            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = slvr.vis_cpu - slvr.bayes_data_cpu
            assert d.shape == (4,slvr.ntime,slvr.nbl,slvr.nchan)

            # Square of the real and imaginary components
            real_term, imag_term = d.real**2, d.imag**2

            # Multiply by the weight vector if required
            if weight_vector is True:
                real_term *= slvr.weight_vector_cpu
                imag_term *= slvr.weight_vector_cpu

            # Reduces a dimension so that we have (nbl,nchan,ntime)
            # (XX.real^2 + XY.real^2 + YX.real^2 + YY.real^2) + 
            # ((XX.imag^2 + XY.imag^2 + YX.imag^2 + YY.imag^2))

            # Sum the real and imaginary terms together
            # for the final result.
            chi_sqrd_terms = np.add.reduce(real_term,axis=0) + \
                np.add.reduce(imag_term,axis=0)
            assert chi_sqrd_terms.shape == (slvr.ntime,slvr.nbl,slvr.nchan)

            return chi_sqrd_terms

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_chi_sqrd(self, weight_vector=False):
        """ Computes the chi squared value.

        Parameters:
            weight_vector : boolean
                True if the chi squared test should be computed with a noise vector

        Returns a floating point scalar values
        """
        slvr = self.solver

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum
        try:
            term_sum = self.compute_chi_sqrd_sum_terms(weight_vector=weight_vector).sum()
            return term_sum if weight_vector is True else term_sum / slvr.sigma_sqrd
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_biro_chi_sqrd(self, weight_vector=False):
        slvr = self.solver
        slvr.vis_cpu = self.compute_ebk_vis()
        return self.compute_chi_sqrd(weight_vector=weight_vector)
