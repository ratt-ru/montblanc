import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from node import *

class Rime2D(Node):
    def __init__(self):
        super(Rime2D, self).__init__()
    def initialise(self, shared_data):
        self.mod = SourceModule("""
#include <pycuda-complex.hpp>
#include \"math_constants.h\"

__device__ void Product2by2(
    const pycuda::complex<double> * lhs,
    const pycuda::complex<double> * rhs,
    pycuda::complex<double> * result)
{
    const pycuda::complex<double> & a00 = lhs[0];
    const pycuda::complex<double> & a10 = lhs[2];
    const pycuda::complex<double> & a01 = lhs[1];
    const pycuda::complex<double> & a11 = lhs[3];

    const pycuda::complex<double> & b00 = rhs[0];
    const pycuda::complex<double> & b10 = rhs[2];
    const pycuda::complex<double> & b01 = rhs[1];
    const pycuda::complex<double> & b11 = rhs[3];

    result[0]=a00*b00+a01*b10;
    result[1]=a00*b01+a01*b11;
    result[2]=a10*b00+a11*b10;
    result[3]=a10*b01+a11*b11;
}

__device__ void Product2by2H(
    const pycuda::complex<double> * lhs,
    const pycuda::complex<double> * rhs,
    pycuda::complex<double> * result)
{
    const pycuda::complex<double> & a00 = lhs[0];
    const pycuda::complex<double> & a10 = lhs[2];
    const pycuda::complex<double> & a01 = lhs[1];
    const pycuda::complex<double> & a11 = lhs[3];

    const pycuda::complex<double> b00 = pycuda::conj(rhs[0]);
    const pycuda::complex<double> b10 = pycuda::conj(rhs[2]);
    const pycuda::complex<double> b01 = pycuda::conj(rhs[1]);
    const pycuda::complex<double> b11 = pycuda::conj(rhs[3]);

    result[0]=a00*b00+a01*b10;
    result[1]=a00*b01+a01*b11;
    result[2]=a10*b00+a11*b10;
    result[3]=a10*b01+a11*b11;
}

__global__ void predict(
    pycuda::complex<double> * VisIn,
    double * UVWin,
    double * LM,
    long * A0,
    long * A1,
    double * wavelength,
    pycuda::complex<double> * jones,
    int ndir, int nchan, int na, int nrows)
{
    // Our space of visibilities is a 3D matrix of BL x DDE x CHAN
    // This is the output

/*
    const unsigned long long int blockId
        = blockIdx.x
        + blockIdx.y*gridDim.x
        + blockIdx.z*gridDim.x*gridDim.y;

    const unsigned long long int threadId
        = blockId*blockDim.x + threadIdx.x;
*/

    #define CUDA_XDIM blockDim.x*gridDim.x
    #define CUDA_YDIM blockDim.y*gridDim.y
    #define CUDA_ZDIM blockDim.z*gridDim.z

    #define SLICE_STRIDE CUDA_YDIM*CUDA_ZDIM
    #define ROW_STRIDE CUDA_ZDIM

    // Baseline
    const int BL = blockIdx.x*blockDim.x + threadIdx.x;
    // Direction Dependent Effect
    const int DDE = blockIdx.y*blockDim.y + threadIdx.y;
    // Channel/Frequency
    const int CHAN = blockIdx.z*blockDim.z + threadIdx.z;

    if(BL >= nrows || CHAN >= nchan || DDE >= ndir)
        return;

    // Constants
    const pycuda::complex<double> I = pycuda::complex<double>(0.,1.);
    const pycuda::complex<double> c0 = 2.0*CUDART_PI*I;
    const double refwave = 1e6;

    // Coalesced loads should occur here!
    // l, m etc. are spaced ndir doubles apart
    // within LM
    // TODO this won't work because
    // DDE's aren't next to each other in the thread
    // sense, instead, CHANS are...
    // WAIT, it might still work, we should get broadcasts...
    double l = LM[DDE+0*ndir];
    double m = LM[DDE+1*ndir];
    double fI = LM[DDE+2*ndir];
    double alpha = LM[DDE+3*ndir];
    double fQ = LM[DDE+4*ndir];
    double fU = LM[DDE+5*ndir];
    double fV = LM[DDE+6*ndir];

    double n = sqrt(1.0 - l*l - m*m) - 1.0;

    pycuda::complex<double> sky[4] =
    {
        fI+fQ,
        fU+I*fV,
        fU-I*fV,
        fI-fQ
    };

    // Coalesced load should occur here!
    // u, v and w are spaced na doubles apart
    double u = UVWin[BL+0*nrows];
    double v = UVWin[BL+1*nrows];
    double w = UVWin[BL+2*nrows];

    double phase = u*l + v*m + w*n;

    return;

    pycuda::complex<double> c1 = c0/wavelength[CHAN];
    double flux = pow(refwave/wavelength[CHAN],alpha);
    pycuda::complex<double> result = flux*pycuda::exp(c1*phase);

    // Index into the visibility matrix
    const int i = (BL*SLICE_STRIDE + DDE*ROW_STRIDE + CHAN)*4; 

    // Our space of jone's matrices is a 3D matrix of ANTENNA x DDE x CHAN
    // This is our input. We choose ANTENNA as our major axis.
    const pycuda::complex<double> * ant0_jones = jones +
        (A0[BL]*SLICE_STRIDE + DDE*ROW_STRIDE + CHAN)*4;
    const pycuda::complex<double> * ant1_jones = jones +
        (A1[BL]*SLICE_STRIDE + DDE*ROW_STRIDE + CHAN)*4;

    pycuda::complex<double> result_jones[4];

    // Internals of Product2by2 should produce coalesced loads
    Product2by2(ant0_jones, sky, result_jones);
    Product2by2H(result_jones, ant1_jones, result_jones);

#if 1
    VisIn[i+0] = result_jones[0]*result;
    VisIn[i+1] = result_jones[1]*result;
    VisIn[i+2] = result_jones[2]*result;
    VisIn[i+3] = result_jones[3]*result;

    VisIn[i+0] = result;
    VisIn[i+1] = result;
    VisIn[i+2] = result;
    VisIn[i+3] = result;

#endif

#if 0
    // Useful for testing that the right indices
    // end up in the right place
    VisIn[i+0] = pycuda::complex<double>(BL,nrows);
    VisIn[i+1] = pycuda::complex<double>(DDE,ndir);
    VisIn[i+2] = pycuda::complex<double>(CHAN,nchan);
    VisIn[i+3] = pycuda::complex<double>(i,0);
#endif

    #undef SLICE_STRIDE
    #undef ROW_STRIDE

    #undef CUDA_XDIM
    #undef CUDA_YDIM
    #undef CUDA_ZDIM
}
""")
        self.kernel = self.mod.get_function('predict')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        ## Here I define my data, and my Jones matrices
        na=10        # Number of antenna
        nrow=10      # Number of rows
        nchan=10     # Number of channels
        ndir=20      # Number of directions

        # Visibilities ! has to have double complex
        #Vis=np.complex128(np.zeros((nrow,nchan,4)))
        # BASELINE x DDE x CHAN
        Vis=np.complex128(np.zeros((nrow,ndir,nchan,4)))
        # UVW coordinates
        uvw=np.float64(np.arange(nrow*3).reshape((nrow,3)))

        # Frequencies in Hz
        freqs=np.float64(np.linspace(1e6,2e6,nchan))
        wavelength=3e8/freqs
        # Sky coordinates
        l=np.float64(np.random.randn(ndir)*0.1)
        m=np.float64(np.random.randn(ndir)*0.1)
        fI=np.float64(np.ones((ndir,)))
        alpha=np.float64(np.zeros((ndir,)))
        fV=np.float64(np.ones((ndir,)))
        fU=np.float64(np.ones((ndir,)))
        fQ=np.float64(np.ones((ndir,)))
        lms=(np.array([l,m,fI,alpha,fV,fU,fQ]).T).copy().astype(np.float64)

        # Antennas
        A0=np.int64(np.random.rand(nrow)*na)
        A1=np.int64(np.random.rand(nrow)*na)

        print
        print A0
        print A1

        # Jones matrices
        #Sols=np.complex128(np.random.randn(ndir,nchan,na,4)+1j*np.random.randn(ndir,nchan,na,4))
        # ANTENNA X DDE X CHAN
        Sols=np.complex128(np.ones((na,ndir,nchan,4))+1j*np.zeros((na,ndir,nchan,4)))

        # Matrix containing information, here just the reference frequency
        # to estimate the flux from spectral index
        Info=np.array([1e6],np.float64)

#        P1=predict.predictSols(Vis, A0, A1, uvw, lms, WaveL, Sols, Info)

        vis_gpu = gpuarray.to_gpu_async(Vis, stream=shared_data.stream[0])
        # GPU CHANGE transpose for the GPU version
        uvw_gpu = gpuarray.to_gpu_async(uvw.T, stream=shared_data.stream[0])
        lms_gpu = gpuarray.to_gpu_async(lms, stream=shared_data.stream[0])
        A0_gpu = gpuarray.to_gpu_async(A0, stream=shared_data.stream[0])
        A1_gpu = gpuarray.to_gpu_async(A1, stream=shared_data.stream[0])
        wavelength_gpu = gpuarray.to_gpu_async(wavelength, stream=shared_data.stream[0])
        sols_gpu = gpuarray.to_gpu_async(Sols, stream=shared_data.stream[0])

        self.kernel(vis_gpu, uvw_gpu, lms_gpu,
            A0_gpu, A1_gpu, wavelength_gpu, sols_gpu,
            np.int32(ndir), np.int32(nchan),
            np.int32(na), np.int32(nrow),
            stream=shared_data.stream[0], block=(8,8,8), grid=(2,2,2))

        vis = vis_gpu.get_async(stream=shared_data.stream[0])

        print vis.shape

        f = open('test.txt','w')

        for v in vis:
            f.write(str(v) + '\n')

        f.close()

    def post_execution(self, shared_data):
        pass
