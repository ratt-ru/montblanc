import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from node import SharedData,ArrayData,Parameter

class BaseSharedData(SharedData):
    na = Parameter(7)
    nbl = Parameter((7**2 + 7)/2)
    nchan = Parameter(8)
    ntime = Parameter(5)
    nsrc = Parameter(10)

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
        self.nbl = nbl = (self.na**2 + self.na)/2
        self.nchan = nchan
        self.ntime = ntime
        self.nsrc = nsrc

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
        self.lm_shape = (2, nsrc)
        self.brightness_shape = (5, nsrc)
        self.wavelength_shape = (nchan)
        self.point_errors_shape = (2, na, ntime)
        self.bayes_model_shape = (4,nbl,nchan,ntime)

        # Set up output data shapes
        self.jones_shape = (4,nbl,nchan,ntime,nsrc)
        self.vis_shape = self.bayes_model_shape
        self.chi_sqrd_result_shape = (nbl,nchan,ntime)

class GPUSharedData(BaseSharedData):
    uvw_gpu = ArrayData()
    lm_gpu = ArrayData()
    brightness_gpu = ArrayData()
    wavelength_gpu = ArrayData()
    point_errors_gpu = ArrayData()
    bayes_model_gpu = ArrayData()

    jones_gpu = ArrayData()
    vis_gpu = ArrayData()
    chi_sqrd_result_gpu = ArrayData()

    def __init__(self, na=7, nchan=8, ntime=5, nsrc=10, dtype=np.float32):
        super(GPUSharedData, self).__init__(na,nchan,ntime,nsrc,dtype)

        # Create the input data arrays on the GPU
        self.uvw_gpu = gpuarray.empty(shape=self.uvw_shape,dtype=self.ft)
        self.lm_gpu = gpuarray.empty(shape=self.lm_shape,dtype=self.ft)
        self.brightness_gpu = gpuarray.empty(shape=self.brightness_shape,dtype=self.ft)
        self.wavelength_gpu = gpuarray.empty(shape=self.wavelength_shape,dtype=self.ft)
        self.point_errors_gpu = gpuarray.empty(shape=self.point_errors_shape,dtype=self.ft)
        self.bayes_model_gpu = gpuarray.empty(shape=self.bayes_model_shape,dtype=self.ct)

        # Create the output data arrays on the GPU
        self.jones_gpu = gpuarray.empty(shape=self.jones_shape,dtype=self.ct)
        self.vis_gpu = gpuarray.empty(shape=self.vis_shape,dtype=self.ct)
        self.chi_sqrd_result_gpu = gpuarray.empty(shape=self.chi_sqrd_result_shape,dtype=self.ft)

        self.gpu_data = [
            self.uvw_gpu,
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