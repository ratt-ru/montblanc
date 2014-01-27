import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class RimeJonesBK(Node):
    def __init__(self):
        super(RimeJonesBK, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule("""
#include <pycuda-complex.hpp>
#include \"math_constants.h\"

extern __shared__ double smem_d[];

__global__
void rime_jones_BK(
    double * UVW,
    double * LMA,
    double * sky,
    double wavelength,
    pycuda::complex<double> * jones,
    int ndir, int na, int nbl)
{
    // Our data space a 2D matrix of BL x DDE

    // Baseline
    const int BL = blockIdx.x*blockDim.x + threadIdx.x;
    // Direction Dependent Effect
    const int DDE = blockIdx.y*blockDim.y + threadIdx.y;

    if(BL >= nbl || DDE >= ndir)
        return;

    // Index into the jones array
    const int i = (BL*ndir + DDE)*4; 

    /* Cache input and output data from global memory. */
    double * u = smem_d;
    double * v = &u[blockDim.x];
    double * w = &v[blockDim.x];
    double * l = &w[blockDim.x];
    double * m = &l[blockDim.y];
    double * a = &m[blockDim.y];

    if (BL < nbl && threadIdx.y == 0)
    {
        u[threadIdx.x] = UVW[BL+0*nbl];
        v[threadIdx.x] = UVW[BL+1*nbl];
        w[threadIdx.x] = UVW[BL+2*nbl];
    }

    if (DDE < ndir && threadIdx.x == 0)
    {
        l[threadIdx.y] = LMA[DDE+0*ndir];
        m[threadIdx.y] = LMA[DDE+1*ndir];
        a[threadIdx.y] = LMA[DDE+2*ndir];
    }

    __syncthreads();

    // Calculate the n term first
    double phase = 1.0 - l[threadIdx.y]*l[threadIdx.y];
    phase -= m[threadIdx.y]*m[threadIdx.y];
    phase = sqrt(phase);

    // u*l + v*m + w*n, in the wrong order :)
    phase *= w[threadIdx.x];                  // w*n
    phase += v[threadIdx.x]*m[threadIdx.y];   // v*m
    phase += u[threadIdx.x]*l[threadIdx.y];   // u*l

    // Multiply by 2*pi/wavelength
    phase *= (2. * CUDART_PI);
    phase /= wavelength;

    // Calculate the complex exponential from the phase
//    pycuda::complex<double> result = pycuda::exp(pycuda::complex<double>(0,phase));
    // Uses two registers less than the above approach.
    pycuda::complex<double> result;
    sincos(phase, &result._M_re, &result._M_im);

    // Multiply by the wavelength to the power of alpha
    result *= pow(1e6/wavelength,a[threadIdx.y]);

#if 0
    // Coalesced store of the computation
    jones[i+0]=pycuda::complex<double>(l[threadIdx.y],u[threadIdx.x]);
    jones[i+1]=pycuda::complex<double>(m[threadIdx.y],v[threadIdx.x]);
    jones[i+2]=pycuda::complex<double>(0.0,w[threadIdx.x]);
    jones[i+3]=pycuda::complex<double>(i,0);
#endif


#if 1
    const double fI = sky[DDE+0*ndir];
    const double fQ = sky[DDE+1*ndir];
    const double fU = sky[DDE+2*ndir];
    const double fV = sky[DDE+3*ndir];
    // TODO, this is uncoalesced
    jones[i+0]=pycuda::complex<double>(fI+fQ,0.)*result;
    jones[i+1]=pycuda::complex<double>(fU,fV)*result;
    jones[i+2]=pycuda::complex<double>(fU,-fV)*result;
    jones[i+3]=pycuda::complex<double>(fU-fQ,0.)*result;

#endif

    #undef SLICE_STRIDE
    #undef ROW_STRIDE

    #undef CUDA_XDIM
    #undef CUDA_YDIM
    #undef CUDA_ZDIM
}
""",
options=['-lineinfo'])
        self.kernel = self.mod.get_function('rime_jones_BK')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        ## Here I define my data, and my Jones matrices
        na=10                    # Number of antenna
        nbl=(na*(na-1))/2        # Number of baselines
        nchan=32                 # Number of channels
        ndir=2048                # Number of DDES

        # Baseline coordinates in the u,v,w (frequency) domain
        uvw_shape = (3,nbl)
        uvw = cuda.pagelocked_empty(uvw_shape,dtype=np.float64)
        uvw[:] = np.array([np.ones(nbl)*1., np.ones(nbl)*2., np.ones(nbl)*3.],dtype=uvw.dtype.type)

        # Frequencies in Hz
        freqs=np.float64(np.linspace(1e6,2e6,nchan))
        wavelength = cuda.pagelocked_empty(freqs.shape,freqs.dtype.type)
        wavelength[:] = 3e8/freqs

        # DDE source coordinates in the l,m,n (sky image) domain
        l=np.float64(np.random.random(ndir)*0.5)
        m=np.float64(np.random.random(ndir)*0.5)
        alpha=np.float64(np.ones((ndir,)))
        lma_shape = (len([l,m,alpha]),ndir)
        lma = cuda.pagelocked_empty(lma_shape,dtype=l.dtype.type)
        lma[:]=np.array([l,m,alpha],dtype=np.float64)

        # Brightness matrix for the DDE sources
        fI=np.float64(np.ones((ndir,)))
        fV=np.float64(np.ones((ndir,)))
        fU=np.float64(np.ones((ndir,)))
        fQ=np.float64(np.ones((ndir,)))
        sky_shape = (len([fI,fV,fU,fQ]),ndir)
        sky = cuda.pagelocked_empty(sky_shape,dtype=fI.dtype.type)
        sky[:] = np.array([fI,fV,fU,fQ], dtype=fI.dtype.type)

        # Output jones matrix
        jones_shape = (nbl,ndir,4)
        jones = cuda.pagelocked_empty(jones_shape, dtype=np.complex128)

        baselines_per_block = 8 if nbl > 8 else nbl
        ddes_per_block = 16 if ndir > 16 else ndir

        baseline_blocks = (nbl + baselines_per_block - 1) / baselines_per_block
        dde_blocks = (ndir + ddes_per_block - 1) / ddes_per_block

        block=(baselines_per_block,ddes_per_block,1)
        grid=(baselines_per_block,ddes_per_block,1)

        print 'block', block, 'grid', grid

        foreground_stream,background_stream = shared_data.stream[0], shared_data.stream[1]
        chan = 0

        uvw_gpu = gpuarray.to_gpu_async(uvw, stream=foreground_stream)
        lma_gpu = gpuarray.to_gpu_async(lma, stream=foreground_stream)
        sky_gpu = gpuarray.to_gpu_async(sky, stream=foreground_stream)
        jones_gpu = gpuarray.empty(jones_shape,dtype=jones.dtype.type)

        self.kernel(uvw_gpu, lma_gpu, sky_gpu,
            wavelength[chan],  jones_gpu,
            np.int32(ndir), np.int32(na), np.int32(nbl),
            stream=foreground_stream,
            block=block,
            grid=grid,
            shared=3*(baselines_per_block+ddes_per_block)
				*np.dtype(np.float64).itemsize)

        print jones_gpu.get_async(stream=foreground_stream)

        shared_data.jones_gpu = jones_gpu
        shared_data.na = na
        shared_data.nbl = nbl
        shared_data.nchan = nchan
        shared_data.ndir = ndir

    def post_execution(self, shared_data):
        pass
