import numpy as np

def rethrow_attribute_exception(e):
    raise AttributeError, '%s. The appropriate numpy array has not ' \
        'been set on the shared data object. You need to set ' \
        'store_cpu=True on your shared data object ' \
        'as well as call the transfer_* method for this to work.' % e

class RimeCPU(object):
    def __init__(self, shared_data):
        self.shared_data = shared_data

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (nbl, ntime, ngsrc, nchan) matrix of floating point scalars.
        """

        sd = self.shared_data

        # The flattened antenna pair array will look something like this.
        # It is based on 2 x nbl x ntime. Here we have 3 baselines and
        # 4 timesteps.
        #
        #            timestep
        #       0 1 2 3 0 1 2 3 0 1 2 3
        #
        # ant1: 0 0 0 0 0 0 0 0 1 1 1 1
        # ant2: 1 1 1 1 2 2 2 2 2 2 2 2

        # Create indexes into the scalar EK terms from the antenna pairs.
        # Scalar EK is 2 x na x ntime x nsrc x nchan.
        ap = np.int32(np.triu_indices(sd.na,1))

        ant1 = np.repeat(ap[0],sd.ntime)*sd.ntime + \
            np.tile(np.arange(sd.ntime), sd.nbl)
        ant2 = np.repeat(ap[1],sd.ntime)*sd.ntime + \
            np.tile(np.arange(sd.ntime), sd.nbl)

        try:
            uvw = sd.uvw_cpu.reshape(3,sd.na*sd.ntime)
            u = (uvw[0][ant1] - uvw[0][ant2]).reshape(sd.nbl, sd.ntime)
            v = (uvw[1][ant1] - uvw[1][ant2]).reshape(sd.nbl, sd.ntime)
            w = (uvw[2][ant1] - uvw[2][ant2]).reshape(sd.nbl, sd.ntime)

            el = sd.gauss_shape_cpu[0]
            em = sd.gauss_shape_cpu[1]

            # OK, try obtain the same results with the fwhm factored out!
            # u1 = u*em - v*el
            # v1 = u*el + v*em
            u1 = (np.outer(u, em) - np.outer(v, el)) \
                .reshape(sd.nbl,sd.ntime,sd.ngsrc)
            v1 = (np.outer(u, el) + np.outer(v, em)) \
                .reshape(sd.nbl,sd.ntime,sd.ngsrc)

            # Obvious given the above reshape
            assert u1.shape == (sd.nbl, sd.ntime, sd.ngsrc)
            assert v1.shape == (sd.nbl, sd.ntime, sd.ngsrc)

            # Construct the scaling factor, this includes the wavelength/frequency
            # into the mix.
            scale_uv = sd.gauss_scale/sd.wavelength_cpu
            assert scale_uv.shape == (sd.nchan,)

            # Multiply u1 and v1 by the scaling factor
            u1 = u1[:,:,:,np.newaxis]*scale_uv[np.newaxis,np.newaxis,np.newaxis,:]
            v1 = v1[:,:,:,np.newaxis]*scale_uv[np.newaxis,np.newaxis,np.newaxis,:]
            # u1 *= R, the ratio of the gaussian axis
            u1 *= sd.gauss_shape_cpu[2][np.newaxis,np.newaxis,:,np.newaxis]

            return np.exp(-(u1**2 + v1**2))

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_gaussian_shape_with_fwhm(self):
        """
        Compute the shape values for the gaussian sources with fwhm factored in.

        Returns a (nbl, nchan, ntime, ngsrc) matrix of floating point scalars.
        """
        sd = self.shared_data

        try:
            # 1.0/sqrt(e_l^2 + e_m^2).
            fwhm_inv = 1.0/np.sqrt(sd.gauss_shape_cpu[0]**2 + sd.gauss_shape_cpu[1]**2)
            # Vector of ngsrc
            assert fwhm_inv.shape == (sd.ngsrc,)

            cos_pa = sd.gauss_shape_cpu[1]*fwhm_inv    # em / fwhm
            sin_pa = sd.gauss_shape_cpu[0]*fwhm_inv    # el / fwhm

            # u1 = u*cos_pa - v*sin_pa
            # v1 = u*sin_pa + v*cos_pa
            u1 = (np.outer(sd.uvw_cpu[0],cos_pa) - np.outer(sd.uvw_cpu[1],sin_pa))\
                .reshape(sd.nbl,sd.ntime,sd.ngsrc)
            v1 = (np.outer(sd.uvw_cpu[0],sin_pa) + np.outer(sd.uvw_cpu[1],cos_pa))\
                .reshape(sd.nbl,sd.ntime,sd.ngsrc)

            # Obvious given the above reshape
            assert u1.shape == (sd.nbl, sd.ntime, sd.ngsrc)
            assert v1.shape == (sd.nbl, sd.ntime, sd.ngsrc)

            # Construct the scaling factor, this includes the wavelength/frequency
            # into the mix.
            scale_uv = sd.gauss_scale/(sd.wavelength_cpu[:,np.newaxis]*fwhm_inv)
            # Should produce nchan x ngsrc
            assert scale_uv.shape == (sd.nchan, sd.ngsrc)

            # u1 *= R, the ratio of the gaussian axis
            u1 *= sd.gauss_shape_cpu[2][np.newaxis,np.newaxis,:]
            # Multiply u1 and v1 by the scaling factor
            u1 = u1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]
            v1 = v1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]

            assert u1.shape == (sd.nbl, sd.nchan, sd.ntime, sd.ngsrc)
            assert v1.shape == (sd.nbl, sd.nchan, sd.ntime, sd.ngsrc)

            return np.exp(-(u1**2 + v1**2))

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_k_jones_scalar_per_ant(self):
        """
        Computes the scalar K (phase) term of the RIME using numpy.

        Returns a (na,ntime,nsrc,nchan) matrix of complex scalars.
        """
        sd = self.shared_data

        try:

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x na.
            n = np.sqrt(1. - sd.lm_cpu[0]**2 - sd.lm_cpu[1]**2) - 1.

            # u*l+v*m+w*n. Outer product creates array of dim na x ntime x nsrcs
            phase = (np.outer(sd.uvw_cpu[0], sd.lm_cpu[0]) + \
                np.outer(sd.uvw_cpu[1], sd.lm_cpu[1]) + \
                np.outer(sd.uvw_cpu[2],n))\
                    .reshape(sd.na, sd.ntime, sd.nsrc)
            assert phase.shape == (sd.na, sd.ntime, sd.nsrc)            

            # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. na x ntime x nchan x nsrcs 
            phase = (2*np.pi*1j*phase)[:,:,:,np.newaxis]/ \
                sd.wavelength_cpu[np.newaxis,np.newaxis,np.newaxis,:]
            assert phase.shape == (sd.na, sd.ntime, sd.nsrc, sd.nchan)

            # Dimension ntime x nsrc x nchan
            power = np.power(sd.ref_wave/sd.wavelength_cpu[np.newaxis,np.newaxis,:],
                sd.brightness_cpu[4,:,:,np.newaxis])
            assert power.shape == (sd.ntime,sd.nsrc,sd.nchan)

            # Combine the power and phase together. Broadcast
            # just works here
            phase_term = power*np.exp(phase)
            assert phase_term.shape == (sd.na, sd.ntime, sd.nsrc, sd.nchan)

            return phase_term

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_k_jones_scalar(self):
        """
        Computes the scalar K (phase) term of the RIME using numpy.

        Returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        try:
            # Repeat the wavelengths along the timesteps for now
            # dim nchan x ntime. 
            w = np.repeat(sd.wavelength_cpu,sd.ntime).reshape(sd.nchan, sd.ntime)

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
            n = np.sqrt(1. - sd.lm_cpu[0]**2 - sd.lm_cpu[1]**2) - 1.

            # u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
            phase = (np.outer(sd.uvw_cpu[0], sd.lm_cpu[0]) + \
                np.outer(sd.uvw_cpu[1], sd.lm_cpu[1]) + \
                np.outer(sd.uvw_cpu[2],n))\
                    .reshape(sd.nbl, sd.ntime, sd.nsrc)
            assert phase.shape == (sd.nbl, sd.ntime, sd.nsrc)            

            # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
            phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
            assert phase.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)            

            # Dim nchan x ntime x nsrcs 
            power = np.power(sd.ref_wave/w[:,:,np.newaxis],
                sd.brightness_cpu[4,np.newaxis,:,:])
            assert power.shape == (sd.nchan, sd.ntime, sd.nsrc)            

            # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
            phase_term = power*np.exp(phase)
            assert phase_term.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)            

            # Multiply the gaussian sources by their shape terms.
            if sd.ngsrc > 0:
                phase_term[:,:,:,sd.npsrc:sd.nsrc] *= self.compute_gaussian_shape()

            return phase_term

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_e_jones_scalar_per_ant(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME per antenna.

        returns a (na,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        try:
            # Compute the offsets for different antenna
            # Broadcasting here produces, na x ntime x nsrc
            l_diff = sd.lm_cpu[0] - sd.point_errors_cpu[0,:,:,np.newaxis]
            m_diff = sd.lm_cpu[1] - sd.point_errors_cpu[1,:,:,np.newaxis]
            E_p = np.sqrt(l_diff**2 + m_diff**2)

            assert E_p.shape == (sd.na, sd.ntime, sd.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_p = sd.beam_width*1e-9*E_p[:,:,:,np.newaxis] *\
                sd.wavelength_cpu[np.newaxis,np.newaxis,np.newaxis,:]
            np.clip(E_p, np.finfo(sd.ft).min, sd.beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (sd.na, sd.ntime, sd.nsrc, sd.nchan)

            return E_p
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_e_jones_scalar(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME

        returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        try:
            # Here we obtain our antenna pairs and pointing errors
            # TODO: The last dimensions are flattened to make indexing easier
            # later. There may be a more numpy way to do this but YOLO.
            ap = sd.get_default_ant_pairs().reshape(2,sd.nbl*sd.ntime)
            pe = sd.point_errors_cpu.reshape(2,sd.na*sd.ntime)

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
            ant0 = ap[0]*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)
            ant1 = ap[1]*sd.ntime + np.tile(np.arange(sd.ntime), sd.nbl)

            # Get the pointing errors for antenna p and q.
            d_p = pe[:,ant0].reshape(2,sd.nbl,sd.ntime)
            d_q = pe[:,ant1].reshape(2,sd.nbl,sd.ntime)

            # Compute the offsets for antenna 0 or p
            # Broadcasting here produces, nbl x ntime x nsrc
            l_off = sd.lm_cpu[0] - d_p[0,:,:,np.newaxis]
            m_off = sd.lm_cpu[1] - d_p[1,:,:,np.newaxis]
            E_p = np.sqrt(l_off**2 + m_off**2)

            assert E_p.shape == (sd.nbl, sd.ntime, sd.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_p = sd.beam_width*1e-9*E_p[:,np.newaxis,:,:]*\
                sd.wavelength_cpu[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_p, np.finfo(sd.ft).min, sd.beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

            # Compute the offsets for antenna 1 or q
            # Broadcasting here produces, nbl x ntime x nsrc
            l_off = sd.lm_cpu[0] - d_q[0,:,:,np.newaxis]
            m_off = sd.lm_cpu[1] - d_q[1,:,:,np.newaxis]
            E_q = np.sqrt(l_off**2 + m_off**2)

            assert E_q.shape == (sd.nbl, sd.ntime, sd.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_q = sd.beam_width*1e-9*E_q[:,np.newaxis,:,:]*\
                sd.wavelength_cpu[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_q, np.finfo(sd.ft).min, sd.beam_clip, E_q)
            E_q = np.cos(E_q)**3

            assert E_q.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

            return E_p/E_q
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_ek_jones_scalar(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Return a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        return self.compute_k_jones_scalar()*self.compute_e_jones_scalar()

    def compute_ek_jones_scalar_per_ant(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Return a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        return self.compute_k_jones_scalar_per_ant()*self.compute_e_jones_scalar_per_ant()

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,ntime,nsrc) matrix of complex scalars.
        """
        sd = self.shared_data

        try:
            # Create the brightness matrix. Dim 4 x ntime x nsrcs
            B = sd.ct([
                sd.brightness_cpu[0]+sd.brightness_cpu[1] + 0j,     # fI+fQ + 0j
                sd.brightness_cpu[2] + 1j*sd.brightness_cpu[3],     # fU + fV*1j
                sd.brightness_cpu[2] - 1j*sd.brightness_cpu[3],     # fU - fV*1j
                sd.brightness_cpu[0]-sd.brightness_cpu[1] + 0j])    # fI-fQ + 0j
            assert B.shape == (4, sd.ntime, sd.nsrc)

            return B

        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_ebk_vis(self):
        """
        Computes the complex visibilities based on the
        scalar EK term and the 2x2 B term.

        Returns a (4,nbl,ntime,nchan) matrix of complex scalars.
        """
        sd = self.shared_data

        # The flattened antenna pair array will look something like this.
        # It is based on 2 x nbl x ntime. Here we have 3 baselines and
        # 4 timesteps.
        #
        #            timestep
        #       0 1 2 3 0 1 2 3 0 1 2 3
        #
        # ant1: 0 0 0 0 0 0 0 0 1 1 1 1
        # ant2: 1 1 1 1 2 2 2 2 2 2 2 2

        # Create indexes into the scalar EK terms from the antenna pairs.
        # Scalar EK is 2 x na x ntime x nsrc x nchan.
        ap = np.int32(np.triu_indices(sd.na,1))
        tcs = sd.ntime*sd.nchan*sd.nsrc

        ant1 = np.repeat(ap[0],tcs)*tcs + np.tile(np.arange(tcs), sd.nbl)
        ant2 = np.repeat(ap[1],tcs)*tcs + np.tile(np.arange(tcs), sd.nbl)

        ek_scalar = sd.jones_scalar_cpu.ravel()

        per_bl_ek_scalar = (ek_scalar[ant1]/ek_scalar[ant2])\
            .reshape(sd.nbl,sd.ntime,sd.nsrc,sd.nchan)

        # Multiply the gaussian sources by their shape terms.
        if sd.ngsrc > 0:
            per_bl_ek_scalar[:,:,sd.npsrc:,:] *= self.compute_gaussian_shape()

        b_jones = self.compute_b_jones()

        jones = per_bl_ek_scalar[np.newaxis,:,:,:,:]*\
            b_jones[:,np.newaxis,:,:,np.newaxis]
        assert jones.shape == (4,sd.nbl,sd.ntime,sd.nsrc,sd.nchan)

        vis = np.add.reduce(jones,axis=3)
        assert vis.shape == (4,sd.nbl,sd.ntime,sd.nchan)

        return vis

    def compute_chi_sqrd_sum_terms(self, weight_vector=False):
        """
        Computes the terms of the chi squared sum, but does not perform the sum itself.

        Parameters:
            weight_vector : boolean
                True if the chi squared test terms should be computed with a noise vector

        Returns a (nbl,nchan,ntime) matrix of floating point scalars.
        """
        sd = self.shared_data

        try:
            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = sd.vis_cpu - sd.bayes_data_cpu
            assert d.shape == (4, sd.nbl, sd.ntime, sd.nchan)

            # Square of the real and imaginary components
            real_term, imag_term = d.real**2, d.imag**2

            # Multiply by the weight vector if required
            if weight_vector is True:
                real_term *= sd.weight_vector_cpu
                imag_term *= sd.weight_vector_cpu

            # Reduces a dimension so that we have (nbl,nchan,ntime)
            # (XX.real^2 + XY.real^2 + YX.real^2 + YY.real^2) + 
            # ((XX.imag^2 + XY.imag^2 + YX.imag^2 + YY.imag^2))

            # Sum the real and imaginary terms together
            # for the final result.
            chi_sqrd_terms = np.add.reduce(real_term,axis=0) + \
                np.add.reduce(imag_term,axis=0)
            assert chi_sqrd_terms.shape == (sd.nbl, sd.ntime, sd.nchan)

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
        sd = self.shared_data

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum
        try:
            term_sum = self.compute_chi_sqrd_sum_terms(weight_vector=weight_vector).sum()
            return term_sum if weight_vector is True else term_sum / sd.sigma_sqrd
        except AttributeError as e:
            rethrow_attribute_exception(e)

    def compute_biro_chi_sqrd(self, weight_vector=False):
        sd = self.shared_data
        sd.vis_cpu = self.compute_ebk_vis()
        return self.compute_chi_sqrd(weight_vector=weight_vector)
