import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from node import SharedData,ArrayData,Parameter

class BaseSharedData(SharedData):
    """ Class defining the RIME Simulation Parameters. """
    na = Parameter(7)
    nbl = Parameter((7**2 + 7)/2)
    nchan = Parameter(8)
    ntime = Parameter(5)
    nsrc = Parameter(10)
    nvis = Parameter(1)

    def __init__(self, na=7, nchan=8, ntime=5, nsrc=10, dtype=np.float32):
        super(BaseSharedData, self).__init__()
        self.set_params(na,nchan,ntime,nsrc,dtype)

    def set_params(self, na, nchan, ntime, nsrc, dtype=np.float32):
        # Configure our problem dimensions. Number of
        # - antenna
        # - baselines
        # - channels
        # - timesteps
        # - sources
        self.na = na
        self.nbl = nbl = (na*(na-1))/2
        self.nchan = nchan
        self.ntime = ntime
        self.nsrc = nsrc
        self.nvis = nbl*nchan*ntime

        # Configure our floating point and complex types
        if dtype == np.float32:
            self.ct = np.complex64
        elif dtype == np.float64:
            self.ct = np.complex128
        else:
            raise TypeError, 'Must specify either np.float32 or np.float64 for dtype'

        self.ft = dtype

        # Set up input data shapes
        self.uvw_shape = (3, nbl, ntime)
        self.ant_pairs_shape = (2, nbl, ntime)
        self.lm_shape = (2, nsrc)
        self.brightness_shape = (5, nsrc)
        self.wavelength_shape = (nchan)
        self.point_errors_shape = (2, na, ntime)
        self.bayes_model_shape = (4,nbl,nchan,ntime)

        # Set up output data shapes
        self.jones_shape = (4,nbl,nchan,ntime,nsrc)
        self.vis_shape = self.bayes_model_shape
        self.chi_sqrd_result_shape = (nbl,nchan,ntime)

        # Initialise sigma squared term and X2 result
        # with default values
        self.set_sigma_sqrd(1.0)
        self.set_X2(0.0)

    def set_ref_freq(self, reffreq):
        """ Set the reference frequency """
        self.refwave = self.ft(reffreq)

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
            "\nAntenna:       " + str(self.na) + \
            "\nBaselines:     " + str(self.nbl) + \
            "\nChannels:      " + str(self.nchan) + \
            "\nTimesteps:     " + str(self.ntime) + \
            "\nSources:       " + str(self.nsrc)

class GPUSharedData(BaseSharedData):
    """
    Class extending BaseSharedData to add GPU arrays
    for holding simulation input and output.
    """
    uvw_gpu = ArrayData()
    ant_pairs_gpu = ArrayData()
    lm_gpu = ArrayData()
    brightness_gpu = ArrayData()
    wavelength_gpu = ArrayData()
    point_errors_gpu = ArrayData()
    bayes_model_gpu = ArrayData()

    jones_gpu = ArrayData()
    vis_gpu = ArrayData()
    chi_sqrd_result_gpu = ArrayData()

    def __init__(self, na=7, nchan=8, ntime=5, nsrc=10, dtype=np.float32, device=None):
        super(GPUSharedData, self).__init__(na,nchan,ntime,nsrc,dtype)

        if device is None:
            import pycuda.autoinit
            self.device = pycuda.autoinit.device
        else:
            self.device = device

        # Figure out the integer compute cability of the device
        cc_tuple = self.device.compute_capability()
        # np.dot((3,5), (100,10)) = 3*100 + 5*10 = 350 for Kepler
        self.cc = np.int32(np.dot(cc_tuple, (100,10)))

        # Create the input data arrays on the GPU
        self.uvw_gpu = gpuarray.zeros(shape=self.uvw_shape,dtype=self.ft)
        self.ant_pairs_gpu = gpuarray.zeros(shape=self.ant_pairs_shape,dtype=np.int32)
        self.lm_gpu = gpuarray.zeros(shape=self.lm_shape,dtype=self.ft)
        self.brightness_gpu = gpuarray.zeros(shape=self.brightness_shape,dtype=self.ft)
        self.wavelength_gpu = gpuarray.zeros(shape=self.wavelength_shape,dtype=self.ft)
        self.point_errors_gpu = gpuarray.zeros(shape=self.point_errors_shape,dtype=self.ft)
        self.bayes_model_gpu = gpuarray.zeros(shape=self.bayes_model_shape,dtype=self.ct)

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
            self.wavelength_gpu,
            self.point_errors_gpu,
            self.bayes_model_gpu,
            self.jones_gpu,
            self.vis_gpu,
            self.chi_sqrd_result_gpu]

    def gpu_mem(self):
        """ Returns the amount of GPU memory used, in bytes """
        return np.array([a.nbytes for a in self.gpu_data]).sum()

    def transfer_uvw(self,uvw):
        self.uvw_gpu.set(uvw)

    def transfer_ant_pairs(self, ant_pairs):
        self.ant_pairs_gpu.set(ant_pairs)

    def transfer_lm(self,lm):
        self.lm_gpu.set(lm)

    def transfer_brightness(self,brightness):
        self.brightness_gpu.set(brightness)

    def transfer_wavelength(self,wavelength):
        self.wavelength_gpu.set(wavelength)

    def transfer_point_errors(self,point_errors):
        self.point_errors_gpu.set(point_errors)

    def transfer_jones(self,jones):
        self.jones_gpu.set(jones)

    def transfer_vis(self,vis):
        self.vis_gpu.set(vis)        

    def transfer_bayes_model(self,bayes_model):
        self.bayes_model_gpu.set(bayes_model)

    def __str__(self):
        return super(GPUSharedData, self).__str__() + \
            "\nGPU Memory:    " + str(self.gpu_mem() / (1024**2)) + " MB"
