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

        Returns a (nbl, nchan, ntime, ngsrc) matrix of floating point scalars.
        """

        slvr = self.solver

        try:
            u = slvr.uvw_cpu[0]
            v = slvr.uvw_cpu[1]
            w = slvr.uvw_cpu[2]
            el = slvr.gauss_shape_cpu[0]
            em = slvr.gauss_shape_cpu[1]
            R = slvr.gauss_shape_cpu[2]

            # OK, try obtain the same results with the fwhm factored out!
            # u1 = u*em - v*el
            # v1 = u*el + v*em
            u1 = (np.outer(u, em) - np.outer(v, el)) \
                .reshape(slvr.nbl,slvr.ntime,slvr.ngsrc)
            v1 = (np.outer(u, el) + np.outer(v, em)) \
                .reshape(slvr.nbl,slvr.ntime,slvr.ngsrc)

            # Obvious given the above reshape
            assert u1.shape == (slvr.nbl, slvr.ntime, slvr.ngsrc)
            assert v1.shape == (slvr.nbl, slvr.ntime, slvr.ngsrc)

            # Construct the scaling factor, this includes the wavelength/frequency
            # into the mix.
            scale_uv = slvr.gauss_scale/slvr.wavelength_cpu
            # Should produce nchan x 1
            assert scale_uv.shape == (slvr.nchan,)

            # Multiply u1 and v1 by the scaling factor
            u1 = u1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,np.newaxis]
            v1 = v1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,np.newaxis]
            # u1 *= R, the ratio of the gaussian axis
            u1 *= R[np.newaxis,np.newaxis,np.newaxis,:]

            return np.exp(-(u1**2 + v1**2))

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_gaussian_shape_with_fwhm(self):
        """
        Compute the shape values for the gaussian sources with fwhm factored in.

        Returns a (nbl, nchan, ntime, ngsrc) matrix of floating point scalars.
        """
        slvr = self.solver

        try:
            # 1.0/sqrt(e_l^2 + e_m^2).
            fwhm_inv = 1.0/np.sqrt(slvr.gauss_shape_cpu[0]**2 + slvr.gauss_shape_cpu[1]**2)
            # Vector of ngsrc
            assert fwhm_inv.shape == (slvr.ngsrc,)

            cos_pa = slvr.gauss_shape_cpu[1]*fwhm_inv    # em / fwhm
            sin_pa = slvr.gauss_shape_cpu[0]*fwhm_inv    # el / fwhm

            # u1 = u*cos_pa - v*sin_pa
            # v1 = u*sin_pa + v*cos_pa
            u1 = (np.outer(slvr.uvw_cpu[0],cos_pa) - np.outer(slvr.uvw_cpu[1],sin_pa))\
                .reshape(slvr.nbl,slvr.ntime,slvr.ngsrc)
            v1 = (np.outer(slvr.uvw_cpu[0],sin_pa) + np.outer(slvr.uvw_cpu[1],cos_pa))\
                .reshape(slvr.nbl,slvr.ntime,slvr.ngsrc)

            # Obvious given the above reshape
            assert u1.shape == (slvr.nbl, slvr.ntime, slvr.ngsrc)
            assert v1.shape == (slvr.nbl, slvr.ntime, slvr.ngsrc)

            # Construct the scaling factor, this includes the wavelength/frequency
            # into the mix.
            scale_uv = slvr.gauss_scale/(slvr.wavelength_cpu[:,np.newaxis]*fwhm_inv)
            # Should produce nchan x ngsrc
            assert scale_uv.shape == (slvr.nchan, slvr.ngsrc)

            # u1 *= R, the ratio of the gaussian axis
            u1 *= slvr.gauss_shape_cpu[2][np.newaxis,np.newaxis,:]
            # Multiply u1 and v1 by the scaling factor
            u1 = u1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]
            v1 = v1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]

            assert u1.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.ngsrc)
            assert v1.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.ngsrc)

            return np.exp(-(u1**2 + v1**2))

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_k_jones_scalar(self):
        """
        Computes the scalar K (phase) term of the RIME using numpy.

        Returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            wave = slvr.wavelength_cpu

            u, v, w = slvr.uvw_cpu[0], slvr.uvw_cpu[1], slvr.uvw_cpu[2]
            l, m = slvr.lm_cpu[0], slvr.lm_cpu[1]
            alpha =slvr.brightness_cpu[4]

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
            n = np.sqrt(1. - l**2 - m**2) - 1.

            # u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
            phase = (np.outer(u,l) + np.outer(v,m) + np.outer(w,n))\
                    .reshape(slvr.nbl, slvr.ntime, slvr.nsrc)
            assert phase.shape == (slvr.nbl, slvr.ntime, slvr.nsrc)            

            # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
            phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:] \
                / wave[np.newaxis,:,np.newaxis,np.newaxis]
            assert phase.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.nsrc)            

            # Dim nchan x ntime x nsrcs 
            power = np.power(slvr.ref_wave/wave[:,np.newaxis,np.newaxis],
                alpha[np.newaxis,:,:])
            assert power.shape == (slvr.nchan, slvr.ntime, slvr.nsrc)            

            # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
            phase_term = power*np.exp(phase)
            assert phase_term.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.nsrc)            

            # Multiply the gaussian sources by their shape terms.
            if slvr.ngsrc > 0:
                phase_term[:,:,:,slvr.npsrc:slvr.nsrc] *= self.compute_gaussian_shape()

            return phase_term

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_per_ant_e_jones_scalar(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME per antenna.

        returns a (na,nchan,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Compute the offsets for different antenna
            # Broadcasting here produces, na x ntime x nsrc
            l_off = slvr.lm_cpu[0] - slvr.point_errors_cpu[0,:,:,np.newaxis]
            m_off = slvr.lm_cpu[1] - slvr.point_errors_cpu[1,:,:,np.newaxis]
            E_p = np.sqrt(l_off**2 + m_off**2)

            assert E_p.shape == (slvr.na, slvr.ntime, slvr.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_p = slvr.beam_width*1e-9*E_p[:,np.newaxis,:,:] *\
                slvr.wavelength_cpu[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_p, np.finfo(slvr.ft).min, slvr.beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (slvr.na, slvr.nchan, slvr.ntime, slvr.nsrc)

            return E_p
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_e_jones_scalar(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME

        returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        try:
            # Here we obtain our antenna pairs and pointing errors
            # TODO: The last dimensions are flattened to make indexing easier
            # later. There may be a more numpy way to do this but YOLO.
            ap = slvr.get_default_ant_pairs().reshape(2,slvr.nbl*slvr.ntime)
            pe = slvr.point_errors_cpu.reshape(2,slvr.na*slvr.ntime)

            # The flattened antenna pair array will look something like this.
            # It is based on 2 x nbl x ntime. Here we have 3 baselines and
            # 4 timesteps.
            #
            #            timestep
            #       0 1 2 3 0 1 2 3 0 1 2 3
            #
            # ant0: 0 0 0 0 0 0 0 0 1 1 1 1
            # ant1: 1 1 1 1 2 2 2 2 2 2 2 2

            # Create indexes into the pointing errors from the antenna pairs.
            # Pointing errors is 2 x na x ntime, thus each index will be
            # i = ANT*ntime + TIME. The TIME additions need to be padded by nbl.
            ant0 = ap[0]*slvr.ntime + np.tile(np.arange(slvr.ntime), slvr.nbl)
            ant1 = ap[1]*slvr.ntime + np.tile(np.arange(slvr.ntime), slvr.nbl)

            # Get the pointing errors for antenna p and q.
            d_p = pe[:,ant0].reshape(2,slvr.nbl,slvr.ntime)
            d_q = pe[:,ant1].reshape(2,slvr.nbl,slvr.ntime)

            # Compute the offsets for antenna 0 or p
            # Broadcasting here produces, nbl x ntime x nsrc
            l_off = slvr.lm_cpu[0] - d_p[0,:,:,np.newaxis]
            m_off = slvr.lm_cpu[1] - d_p[1,:,:,np.newaxis]
            E_p = np.sqrt(l_off**2 + m_off**2)

            assert E_p.shape == (slvr.nbl, slvr.ntime, slvr.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_p = slvr.beam_width*1e-9*E_p[:,np.newaxis,:,:]*\
                slvr.wavelength_cpu[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_p, np.finfo(slvr.ft).min, slvr.beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.nsrc)

            # Compute the offsets for antenna 1 or q
            # Broadcasting here produces, nbl x ntime x nsrc
            l_off = slvr.lm_cpu[0] - d_q[0,:,:,np.newaxis]
            m_off = slvr.lm_cpu[1] - d_q[1,:,:,np.newaxis]
            E_q = np.sqrt(l_off**2 + m_off**2)

            assert E_q.shape == (slvr.nbl, slvr.ntime, slvr.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_q = slvr.beam_width*1e-9*E_q[:,np.newaxis,:,:]*\
                slvr.wavelength_cpu[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_q, np.finfo(slvr.ft).min, slvr.beam_clip, E_q)
            E_q = np.cos(E_q)**3

            assert E_q.shape == (slvr.nbl, slvr.nchan, slvr.ntime, slvr.nsrc)

            return E_p/E_q
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_ek_jones_scalar(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Return a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        return self.compute_k_jones_scalar()*self.compute_e_jones_scalar()

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,nsrc) matrix of complex scalars.
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

    def compute_bk_jones(self):
        """
        Computes the BK term of the RIME.

        Returns a (4,nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        slvr = self.solver

        # Compute the K and B terms
        scalar_K = self.compute_k_jones_scalar()
        B = self.compute_b_jones()

        # This works due to broadcast! Multiplies phase and brightness along
        # srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
        jones_cpu = (scalar_K[np.newaxis,:,:,:,:]* \
            B[:,np.newaxis, np.newaxis,:,:])#\
            #.reshape((4, slvr.nbl, slvr.nchan, slvr.ntime, slvr.nsrc))
        assert jones_cpu.shape == slvr.jones_shape

        return jones_cpu 

    def compute_ebk_jones(self):
        """
        Computes the BK term of the RIME.

        Returns a (4,nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        return self.compute_bk_jones()*self.compute_e_jones_scalar()

    def compute_bk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar K term and the 2x2 B term.

        Returns a (4,nbl,nchan,ntime) matrix of complex scalars.
        """
        return np.add.reduce(self.compute_bk_jones(), axis=4)        

    def compute_ebk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,nbl,nchan,ntime) matrix of complex scalars.
        """
        return np.add.reduce(self.compute_ebk_jones(), axis=4)        

    def compute_chi_sqrd_sum_terms(self, weight_vector=False):
        """
        Computes the terms of the chi squared sum, but does not perform the sum itself.

        Parameters:
            weight_vector : boolean
                True if the chi squared test terms should be computed with a noise vector

        Returns a (nbl,nchan,ntime) matrix of floating point scalars.
        """
        slvr = self.solver

        try:
            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = slvr.vis_cpu - slvr.bayes_data_cpu

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
            chi_sqrd_terms = np.add.reduce(real_term,axis=0) + np.add.reduce(imag_term,axis=0)

            assert chi_sqrd_terms.shape == (slvr.nbl, slvr.nchan, slvr.ntime)

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
        slvr.vis_cpu = compute_ebk_vis(slvr)
        return self.compute_chi_sqrd(weight_vector=weight_vector)
