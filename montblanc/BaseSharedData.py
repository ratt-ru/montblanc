import sys
import numpy as np

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc

class Parameter(object):
    """ Unused Descriptor Class. For gpuarrays """
    def __init__(self, value=None):
#        if value is None:
#            value = []

        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Negative parameter value: %s' % value )
        self.value = value


class ArrayData(object):
    """ Unused Descriptor Class. For gpuarrays """
    def __init__(self, value=None):
#        if value is None:
#            value = []

        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = value

def get_nr_of_baselines(na):
    """ Compute the number of baselines for the 
    given number of antenna """
    return (na*(na-1))//2

class SharedData(object):
    """ Base class for data shared amongst pipeline nodes.

    In practice, nodes will be responsible for creating,
    updating and deleting members of this class.
    Its not a complicated beast.
    """
    pass

class BaseSharedData(SharedData):
    """ Class defining the RIME Simulation Parameters. """
    na = Parameter(7)
    nbl = Parameter(get_nr_of_baselines(7))
    nchan = Parameter(8)
    ntime = Parameter(5)
    npsrc = Parameter(10)
    ngsrc = Parameter(10)
    nsrc = Parameter(20)
    nvis = Parameter(1)

    def __init__(self, na=7, nchan=8, ntime=5, npsrc=10, ngsrc=10, dtype=np.float64):
        super(BaseSharedData, self).__init__()
        self.set_params(na,nchan,ntime,npsrc,ngsrc,dtype)

    def set_params(self, na, nchan, ntime, npsrc, ngsrc, dtype=np.float32):
        # Configure our problem dimensions. Number of
        # - antenna
        # - baselines
        # - channels
        # - timesteps
        # - point sources
        # - gaussian sources
        self.na = na
        self.nbl = nbl = get_nr_of_baselines(na)
        self.nchan = nchan
        self.ntime = ntime
        self.npsrc = npsrc
        self.ngsrc = ngsrc
        self.nsrc = nsrc = npsrc + ngsrc
        self.nvis = nbl*nchan*ntime

        if nsrc == 0:
            raise ValueError, 'The number of sources, or, the sum of npsrc and ngsrc, must be greater than zero'

        # Configure our floating point and complex types
        if dtype == np.float32:
            self.ct = np.complex64
        elif dtype == np.float64:
            self.ct = np.complex128
        else:
            raise TypeError, 'Must specify either np.float32 or np.float64 for dtype'

        self.ft = dtype

        # UVW coordinates per baseline, changing over time
        self.uvw_shape = (3, nbl, ntime)
        # Antenna Pairs per baseline, changing over time
        self.ant_pairs_shape = (2, nbl, ntime)

        # Point Sources
        self.lm_shape = (2, nsrc)
        self.brightness_shape = (5, ntime, nsrc)

        # Gaussian Shapes
        self.gauss_shape_shape = (3, ngsrc)

        self.wavelength_shape = (nchan)
        self.point_errors_shape = (2, na, ntime)
        self.bayes_data_shape = (4,nbl,nchan,ntime)

        # Set up output data shapes
        self.jones_shape = (4,nbl,nchan,ntime,nsrc)
        self.vis_shape = self.bayes_data_shape
        self.chi_sqrd_result_shape = (nbl,nchan,ntime)
        self.weight_vector_shape = self.bayes_data_shape

        # Set up gaussian scaling parameters
        # Derived from https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L493
        # and https://github.com/ska-sa/meqtrees-timba/blob/master/MeqNodes/src/PSVTensor.cc#L602
        self.fwhm2int = 1.0/np.sqrt(np.log(256))
        # Note that we don't divide by speed of light here. meqtrees code operates
        # on frequency, while we're dealing with wavelengths.
        self.gauss_scale = self.fwhm2int*np.sqrt(2)*np.pi

        # Initialise sigma squared term and X2 result
        # with default values
        self.set_sigma_sqrd(1.0)
        self.set_X2(0.0)

        # Initialise the cos3 constant
        self.set_beam_width(65)

        # Initialise the beam clipping paramter
        self.set_beam_clip(1.0881)

    def get_params(self):
        sd = self
        
        return {
            'na' : sd.na,
            'nbl' : sd.nbl,
            'nchan' : sd.nchan,
            'ntime' : sd.ntime,
            'npsrc' : sd.npsrc,
            'ngsrc' : sd.ngsrc,
            'nsrc'  : sd.nsrc,
            'nvis' : sd.nvis,
            'beam_width': sd.beam_width,
            'beam_clip' : sd.E_beam_clip,
            'sigma_sqrd' : sd.sigma_sqrd,
            'gauss_scale' : sd.gauss_scale
        }

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def set_beam_width(self, beam_width):
        """
        Set the beam width used in the analytic E term.

        Should be set in metres.

        >>> sd.set_beam_width(65)

        """
        self.beam_width = self.ft(beam_width)

    def set_beam_clip(self, clip):
        """ Set the beam clipping parameter used in the analytic E term """
        self.E_beam_clip = self.ft(clip)

    def set_ref_wave(self, ref_wave):
        """ Set the reference wavelength """
        self.ref_wave = self.ft(ref_wave)

    def set_sigma_sqrd(self, sigma_sqrd):
        """ Set the sigma squared term, used
        for chi squared """
        self.sigma_sqrd = self.ft(sigma_sqrd)

    def set_X2(self, X2):
        """ Set the chi squared result. Useful for sensibly initialising it """
        self.X2 = self.ft(X2)

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, nbl, ntime), dtype=np.int32]) containing the
        default antenna pairs for each baseline at each timestep.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna. 
        sd = self

        tmp = np.int32(np.triu_indices(sd.na,1))
        tmp = np.tile(tmp,sd.ntime).reshape(2,sd.ntime,sd.nbl)
        tmp = np.rollaxis(tmp, axis=2, start=1)
        assert tmp.shape == sd.ant_pairs_shape
        return tmp.copy()

    def __str__(self):
        return "RIME Simulation Dimensions" + \
            "\nAntenna:          " + str(self.na) + \
            "\nBaselines:        " + str(self.nbl) + \
            "\nChannels:         " + str(self.nchan) + \
            "\nTimesteps:        " + str(self.ntime) + \
            "\nPoint Sources:    " + str(self.npsrc) + \
            "\nGaussian Sources: " + str(self.ngsrc) + \
            "\nTotal Sources:    " + str(self.nsrc)

class GPUSharedData(BaseSharedData):
    """
    Class extending BaseSharedData to add GPU arrays
    for holding simulation input and output.
    """
    uvw_gpu = ArrayData()
    ant_pairs_gpu = ArrayData()
    lm_gpu = ArrayData()
    brightness_gpu = ArrayData()
    gauss_shape_gpu = ArrayData()
    wavelength_gpu = ArrayData()
    point_errors_gpu = ArrayData()
    bayes_data_gpu = ArrayData()

    jones_gpu = ArrayData()
    vis_gpu = ArrayData()
    chi_sqrd_result_gpu = ArrayData()
    weight_vector_gpu = ArrayData()

    def __init__(self, na=7, nchan=8, ntime=5, npsrc=10, ngsrc=10, dtype=np.float32, **kwargs):
        """
        GPUSharedData constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
        Keyword Arguments:
            device : pycuda.device.Device
                CUDA device to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSharedData object.
    """
        super(GPUSharedData, self).__init__(na=na,nchan=nchan,ntime=ntime, \
            npsrc=npsrc,ngsrc=ngsrc,dtype=dtype)

        # Store the device, choosing the default if not specified
        self.device = kwargs.get('device')

        if self.device is None:
            import pycuda.autoinit
            self.device = pycuda.autoinit.device

        # Should we store CPU versions of the GPU arrays
        self.store_cpu = kwargs.get('store_cpu', False)

        # Figure out the integer compute cability of the device
        cc_tuple = self.device.compute_capability()
        # np.dot((3,5), (100,10)) = 3*100 + 5*10 = 350 for Kepler
        self.cc = np.int32(np.dot(cc_tuple, (100,10)))

        # Create the input data arrays on the GPU
        self.uvw_gpu = gpuarray.zeros(shape=self.uvw_shape,dtype=self.ft)
        self.ant_pairs_gpu = gpuarray.zeros(shape=self.ant_pairs_shape,dtype=np.int32)

        self.lm_gpu = gpuarray.zeros(shape=self.lm_shape,dtype=self.ft)
        self.brightness_gpu = gpuarray.zeros(shape=self.brightness_shape,dtype=self.ft)

        # We could have zero gaussian sources, in which case PyCUDA falls over trying
        # to fill a zero length array.
        if np.product(self.gauss_shape_shape) > 0:
            self.gauss_shape_gpu = gpuarray.zeros(shape=self.gauss_shape_shape,dtype=self.ft)
        else:
            self.gauss_shape_gpu = gpuarray.empty(shape=self.gauss_shape_shape,dtype=self.ft)

        self.wavelength_gpu = gpuarray.zeros(shape=self.wavelength_shape,dtype=self.ft)
        self.point_errors_gpu = gpuarray.zeros(shape=self.point_errors_shape,dtype=self.ft)
        self.weight_vector_gpu = gpuarray.zeros(shape=self.weight_vector_shape,dtype=self.ft)
        self.bayes_data_gpu = gpuarray.zeros(shape=self.bayes_data_shape,dtype=self.ct)

        # Create the output data arrays on the GPU
        self.jones_gpu = gpuarray.zeros(shape=self.jones_shape,dtype=self.ct)
        self.vis_gpu = gpuarray.zeros(shape=self.vis_shape,dtype=self.ct)
        self.chi_sqrd_result_gpu = gpuarray.zeros(shape=self.chi_sqrd_result_shape,dtype=self.ft)

        # Create a list of the GPU arrays
        self.gpu_data = [
            self.uvw_gpu,
            self.ant_pairs_gpu,
            self.lm_gpu,
            self.brightness_gpu,
            self.gauss_shape_gpu,
            self.wavelength_gpu,
            self.point_errors_gpu,
            self.weight_vector_gpu,
            self.bayes_data_gpu,
            self.jones_gpu,
            self.vis_gpu,
            self.chi_sqrd_result_gpu]

    def check_array(self, name, npary, gpuary):
        if npary.shape != gpuary.shape:
            raise ValueError, '%s\'s shape %s is different from the expected shape %s.' % (name, npary.shape, gpuary.shape)

        if npary.dtype.type != gpuary.dtype.type:
            raise TypeError, 'Type of %s, \'%s\' is different from the expected type %s.' % (name, npary.dtype.type, gpuary.dtype.type)          

    def gpu_mem(self):
        """ Returns the amount of GPU memory used, in bytes """
        return np.array([a.nbytes for a in self.gpu_data]).sum()

    def transfer_uvw(self,uvw):
        self.check_array('uvw', uvw, self.uvw_gpu)
        if self.store_cpu is True: self.uvw=uvw
        self.uvw_gpu.set(uvw)

    def transfer_ant_pairs(self, ant_pairs):
        self.check_array('ant_pairs', ant_pairs, self.ant_pairs_gpu)
        if self.store_cpu is True: self.ant_pairs=ant_pairs
        self.ant_pairs_gpu.set(ant_pairs)

    def transfer_lm(self,lm):
        self.check_array('lm', lm, self.lm_gpu)
        if self.store_cpu is True: self.lm = lm
        self.lm_gpu.set(lm)

    def transfer_brightness(self,brightness):
        self.check_array('brightness', brightness, self.brightness_gpu)
        if self.store_cpu is True: self.brightness = brightness
        self.brightness_gpu.set(brightness)

    def transfer_gauss_shape(self,gauss_shape):
        self.check_array('gauss_shape', gauss_shape, self.gauss_shape_gpu)
        if self.store_cpu is True: self.gauss_shape = gauss_shape
        self.gauss_shape_gpu.set(gauss_shape)

    def transfer_wavelength(self, wavelength):
        self.check_array('wavelength', wavelength, self.wavelength_gpu)
        if self.store_cpu is True: self.wavelength = wavelength
        self.wavelength_gpu.set(wavelength)

    def transfer_point_errors(self,point_errors):
        self.check_array('point_errors', point_errors, self.point_errors_gpu)
        if self.store_cpu is True: self.point_errors = point_errors
        self.point_errors_gpu.set(point_errors)

    def transfer_jones(self,jones):
        self.check_array('jones', jones, self.jones_gpu)
        if self.store_cpu is True: self.jones = jones
        self.jones_gpu.set(jones)

    def transfer_vis(self,vis):
        self.check_array('vis', vis, self.vis_gpu)
        if self.store_cpu is True: self.vis = vis
        self.vis_gpu.set(vis)        

    def transfer_bayes_data(self,bayes_data):
        self.check_array('bayes_data', bayes_data, self.bayes_data_gpu)
        if self.store_cpu is True: self.bayes_data = bayes_data
        self.bayes_data_gpu.set(bayes_data)

    def transfer_weight_vector(self, weight_vector):
        self.check_array('weight_vector', weight_vector, self.weight_vector_gpu)
        if self.store_cpu is True: self.weight_vector = weight_vector
        self.weight_vector_gpu.set(weight_vector)

    def __str__(self):
        return super(GPUSharedData, self).__str__() + \
            "\nGPU Memory:    %.3f MB" % (self.gpu_mem() / (1024.**2))

    def rethrow_attribute_exception(self, e):
        raise AttributeError, '%s. The appropriate numpy array has not ' \
            'been set on the shared data object. You need to set ' \
            'store_cpu=True on your shared data object ' \
            'as well as call the transfer_* method for this to work.' % e

    def compute_gaussian_shape(self):
        """
        Compute the shape values for the gaussian sources.

        Returns a (nbl, nchan, ntime, ngsrc) matrix of floating point scalars.
        """
        sd = self

        # 1.0/sqrt(e_l^2 + e_m^2).
        fwhm_inv = 1.0/np.sqrt(sd.gauss_shape[0]**2 + sd.gauss_shape[1]**2)
        # Vector of ngsrc
        assert fwhm_inv.shape == (sd.ngsrc,)

        cos_pa = sd.gauss_shape[0]*fwhm_inv
        sin_pa = sd.gauss_shape[1]*fwhm_inv

        # u1 = u*cos_pa - v*sin_pa
        # v1 = u*sin_pa + v*cos_pa
        u1 = (np.outer(sd.uvw[0],cos_pa) - np.outer(sd.uvw[1],sin_pa)).reshape(sd.nbl,sd.ntime,sd.ngsrc)
        v1 = (np.outer(sd.uvw[0],sin_pa) + np.outer(sd.uvw[1],cos_pa)).reshape(sd.nbl,sd.ntime,sd.ngsrc)

        # Obvious given the above reshape
        assert u1.shape == (sd.nbl, sd.ntime, sd.ngsrc)
        assert v1.shape == (sd.nbl, sd.ntime, sd.ngsrc)

        # Construct the scaling factor, this includes the wavelength/frequency
        # into the mix.
        scale_uv = self.gauss_scale/(sd.wavelength[:,np.newaxis]*fwhm_inv)
        # Should produce nchan x ngsrc
        assert scale_uv.shape == (sd.nchan, sd.ngsrc)

        # u1 *= R, the ratio of the gaussian axis
        u1 *= sd.gauss_shape[2][np.newaxis,np.newaxis,:]
        # Multiply u1 and v1 by the scaling factor
        u1 = u1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]
        v1 = v1[:,np.newaxis,:,:]*scale_uv[np.newaxis,:,np.newaxis,:]

        assert u1.shape == (sd.nbl, sd.nchan, sd.ntime, sd.ngsrc)
        assert v1.shape == (sd.nbl, sd.nchan, sd.ntime, sd.ngsrc)

        return np.exp(-(u1**2 + v1**2))

    def compute_k_jones_scalar(self):
        """
        Computes the scalar K (phase) term of the RIME using numpy.

        Returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self

        try:
            sd = self
            # Repeat the wavelengths along the timesteps for now
            # dim nchan x ntime. 
            w = np.repeat(sd.wavelength,sd.ntime).reshape(sd.nchan, sd.ntime)

            # n = sqrt(1 - l^2 - m^2) - 1. Dim 1 x nbl.
            n = np.sqrt(1. - sd.lm[0]**2 - sd.lm[1]**2) - 1.

            # u*l+v*m+w*n. Outer product creates array of dim nbl x ntime x nsrcs
            phase = (np.outer(sd.uvw[0], sd.lm[0]) + \
                np.outer(sd.uvw[1], sd.lm[1]) + \
                np.outer(sd.uvw[2],n))\
                    .reshape(sd.nbl, sd.ntime, sd.nsrc)
            assert phase.shape == (sd.nbl, sd.ntime, sd.nsrc)            

            # 2*pi*sqrt(u*l+v*m+w*n)/wavelength. Dim. nbl x nchan x ntime x nsrcs 
            phase = (2*np.pi*1j*phase)[:,np.newaxis,:,:]/w[np.newaxis,:,:,np.newaxis]
            assert phase.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)            

            # Dim nchan x ntime x nsrcs 
            power = np.power(sd.ref_wave/w[:,:,np.newaxis], sd.brightness[4,np.newaxis,:,:])
            assert power.shape == (sd.nchan, sd.ntime, sd.nsrc)            

            # This works due to broadcast! Dim nbl x nchan x ntime x nsrcs
            phase_term = power*np.exp(phase)
            assert phase_term.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)            

            # Multiply the gaussian sources by their shape terms.
            if sd.ngsrc > 0:
                phase_term[:,:,:,sd.npsrc:sd.nsrc] *= self.compute_gaussian_shape()

            return phase_term

        except AttributeError as e:
            self.rethrow_attribute_exception(e)

    def compute_e_jones_scalar(self):
        """
        Computes the scalar E (analytic cos^3) term of the RIME

        returns a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self

        try:
            # Here we obtain our antenna pairs and pointing errors
            # TODO: The last dimensions are flattened to make indexing easier
            # later. There may be a more numpy way to do this but YOLO.
            ap = sd.get_default_ant_pairs().reshape(2,sd.nbl*sd.ntime)
            pe = sd.point_errors.reshape(2,sd.na*sd.ntime)

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
            l_off = sd.lm[0] - d_p[0,:,:,np.newaxis]
            m_off = sd.lm[1] - d_p[1,:,:,np.newaxis]
            E_p = np.sqrt(l_off**2 + m_off**2)

            assert E_p.shape == (sd.nbl, sd.ntime, sd.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_p = sd.beam_width*1e-9*E_p[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_p, np.finfo(sd.ft).min, sd.E_beam_clip, E_p)
            E_p = np.cos(E_p)**3

            assert E_p.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

            # Compute the offsets for antenna 1 or q
            # Broadcasting here produces, nbl x ntime x nsrc
            l_off = sd.lm[0] - d_q[0,:,:,np.newaxis]
            m_off = sd.lm[1] - d_q[1,:,:,np.newaxis]
            E_q = np.sqrt(l_off**2 + m_off**2)

            assert E_q.shape == (sd.nbl, sd.ntime, sd.nsrc)

            # Broadcasting here produces, nbl x nchan x ntime x nsrc
            E_q = sd.beam_width*1e-9*E_q[:,np.newaxis,:,:]*sd.wavelength[np.newaxis,:,np.newaxis,np.newaxis]
            np.clip(E_q, np.finfo(sd.ft).min, sd.E_beam_clip, E_q)
            E_q = np.cos(E_q)**3

            assert E_q.shape == (sd.nbl, sd.nchan, sd.ntime, sd.nsrc)

            return E_p*E_q
        except AttributeError as e:
            self.rethrow_attribute_exception(e)

    def compute_ek_jones_scalar(self):
        """
        Computes the scalar EK (phase*cos^3) term of the RIME.

        Return a (nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        return self.compute_k_jones_scalar()*\
            self.compute_e_jones_scalar()

    def compute_b_jones(self):
        """
        Computes the B term of the RIME.

        Returns a (4,nsrc) matrix of complex scalars.
        """
        sd = self
        try:
            # Create the brightness matrix. Dim 4 x ntime x nsrcs
            B = sd.ct([
                sd.brightness[0]+sd.brightness[1] + 0j,     # fI+fQ + 0j
                sd.brightness[2] + 1j*sd.brightness[3],     # fU + fV*1j
                sd.brightness[2] - 1j*sd.brightness[3],     # fU - fV*1j
                sd.brightness[0]-sd.brightness[1] + 0j])    # fI-fQ + 0j
            assert B.shape == (4, sd.ntime, sd.nsrc)

            return B

        except AttributeError as e:
            self.rethrow_attribute_exception(e)

    def compute_bk_jones(self):
        """
        Computes the BK term of the RIME.

        Returns a (4,nbl,nchan,ntime,nsrc) matrix of complex scalars.
        """
        sd = self

        # Compute the K and B terms
        scalar_K = self.compute_k_jones_scalar()
        B = self.compute_b_jones()

        # This works due to broadcast! Multiplies phase and brightness along
        # srcs axis of brightness. Dim 4 x nbl x nchan x ntime x nsrcs.
        jones_cpu = (scalar_K[np.newaxis,:,:,:,:]* \
            B[:,np.newaxis, np.newaxis,:,:])#\
            #.reshape((4, sd.nbl, sd.nchan, sd.ntime, sd.nsrc))
        assert jones_cpu.shape == sd.jones_shape

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
        sd = self

        try:
            # Take the difference between the visibilities and the model
            # (4,nbl,nchan,ntime)
            d = sd.vis - sd.bayes_data

            # Square of the real and imaginary components
            real_term, imag_term = d.real**2, d.imag**2

            # Multiply by the weight vector if required
            if weight_vector is True:
                real_term *= sd.weight_vector
                imag_term *= sd.weight_vector

            # Reduces a dimension so that we have (nbl,nchan,ntime)
            # (XX.real^2 + XY.real^2 + YX.real^2 + YY.real^2) + 
            # ((XX.imag^2 + XY.imag^2 + YX.imag^2 + YY.imag^2))

            # Sum the real and imaginary terms together
            # for the final result.
            chi_sqrd_terms = np.add.reduce(real_term,axis=0) + np.add.reduce(imag_term,axis=0)

            assert chi_sqrd_terms.shape == (sd.nbl, sd.nchan, sd.ntime)

            return chi_sqrd_terms

        except AttributeError as e:
            self.rethrow_attribute_exception(e)

    def compute_chi_sqrd(self, weight_vector=False):
        """ Computes the chi squared value.

        Parameters:
            weight_vector : boolean
                True if the chi squared test should be computed with a noise vector

        Returns a floating point scalar values
        """
        sd = self

        # Do the chi squared sum on the CPU.
        # If we're not using the weight vector, sum and
        # divide by the sigma squared.
        # Otherwise, simply return the sum
        try:
            term_sum = self.compute_chi_sqrd_sum_terms(weight_vector=weight_vector).sum()
            return term_sum if weight_vector is True else term_sum / sd.sigma_sqrd
        except AttributeError as e:
            self.rethrow_attribute_exception(e)

    def compute_biro_chi_sqrd(self, weight_vector=False):
        self.vis = self.compute_ebk_vis()
        return self.compute_chi_sqrd(weight_vector=weight_vector)
