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
    pycuda::complex<double> * jones,
    double wavelength,
    int ndir, int na, int nbl)
{
    // Our space of visibilities is a 2D matrix of BL x DDE
    // This is the output

    #define CUDA_XDIM blockDim.x*gridDim.x
    #define CUDA_YDIM blockDim.y*gridDim.y
    #define CUDA_ZDIM blockDim.z*gridDim.z

    #define ROW_STRIDE CUDA_YDIM

    // Baseline
    const int BL = blockIdx.x*blockDim.x + threadIdx.x;
    // Direction Dependent Effect
    const int DDE = blockIdx.y*blockDim.y + threadIdx.y;

    if(BL >= nbl || DDE >= ndir)
        return;

    // Index into the visibility matrix
    const int i = (BL*ndir + DDE)*4; 

#if 0
    // Useful for testing that the right indices
    // end up in the right place
    VisIn[i+0] = pycuda::complex<double>(BL,nbl);
    VisIn[i+1] = pycuda::complex<double>(DDE,ndir);
    VisIn[i+2] = pycuda::complex<double>(threadIdx.x,threadIdx.y);
    VisIn[i+3] = pycuda::complex<double>(i,0);
#endif

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

    __syncthreads();

    double n = sqrt(1.0 - l*l - m*m);

    pycuda::complex<double> sky[4] =
    {
        fI+fQ,
        fU+I*fV,
        fU-I*fV,
        fI-fQ
    };

    // Coalesced load should occur here!
    // u, v and w are spaced na doubles apart
    double u = UVWin[BL+0*nbl];
    double v = UVWin[BL+1*nbl];
    double w = UVWin[BL+2*nbl];

    __syncthreads();

    double phase = u*l + v*m + w*n;

    pycuda::complex<double> c1 = c0/wavelength;
    double flux = pow(refwave/wavelength,alpha);
    pycuda::complex<double> result = flux*pycuda::exp(c1*phase);

    // Our space of jone's matrices is a 3D matrix of ANTENNA x DDE x CHAN
    // This is our input. We choose ANTENNA as our major axis.
    const pycuda::complex<double> * ant0_jones = jones +
        (A0[BL]*ndir + DDE)*4;
    const pycuda::complex<double> * ant1_jones = jones +
        (A1[BL]*ndir + DDE)*4;

    pycuda::complex<double> result_jones[4];

    // Internals of Product2by2 should produce coalesced loads
    Product2by2(ant0_jones, sky, result_jones);
    Product2by2H(result_jones, ant1_jones, result_jones);

#if 1
//    VisIn[i+0] = result_jones[0]*result;
//    VisIn[i+1] = result_jones[1]*result;
//    VisIn[i+2] = result_jones[2]*result;
//    VisIn[i+3] = result_jones[3]*result;

    VisIn[i+0] = pycuda::complex<double>(u,l);
    VisIn[i+1] = pycuda::complex<double>(v,m);
    VisIn[i+2] = pycuda::complex<double>(w,n);
    VisIn[i+3] = result;

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
        na=10                   # Number of antenna
        nbl=(na*(na-1))/2     # Number of baselines
        nchan=6                 # Number of channels
        ndir=200                # Number of DDES

        # Visibilities ! has to have double complex
        # vis=np.complex128(np.zeros((nbl,nchan,4)))
        # CHAN x BASELINE x DDE
        vis_shape = (nchan,nbl,ndir,4)
        vis = cuda.pagelocked_empty(vis_shape,np.complex128)
        vis[:] = np.zeros(vis_shape).astype(vis.dtype.type)
        # UVW coordinates
        uvw_shape = (3,nbl)
        uvw = cuda.pagelocked_empty(uvw_shape,np.float64)
        uvw[:] = np.array([np.ones(nbl)*1., np.ones(nbl)*2., np.ones(nbl)*3.],dtype=uvw.dtype.type)

        # Frequencies in Hz
        freqs=np.float64(np.linspace(1e6,2e6,nchan))
        wavelength = cuda.pagelocked_empty(freqs.shape,freqs.dtype.type)
        wavelength[:] = 3e8/freqs

        # Sky coordinates
        l=np.float64(np.random.random(ndir)*0.5)
        m=np.float64(np.random.random(ndir)*0.5)
        fI=np.float64(np.ones((ndir,)))
        alpha=np.float64(np.zeros((ndir,)))
        fV=np.float64(np.ones((ndir,)))
        fU=np.float64(np.ones((ndir,)))
        fQ=np.float64(np.ones((ndir,)))
        lms_shape = (len([l,m,fI,alpha,fV,fU,fQ]),ndir)
        lms = cuda.pagelocked_empty(lms_shape,l.dtype.type)
        lms[:]=(np.array([l,m,fI,alpha,fV,fU,fQ])).astype(np.float64)

        # Antennas
        A0=cuda.pagelocked_empty((nbl), np.int64)
        A1=cuda.pagelocked_empty((nbl), np.int64)
        A0[:]=np.random.rand(nbl)*na
        A1[:]=np.random.rand(nbl)*na

        # Jones matrices
        #Sols=np.complex128(np.random.randn(ndir,nchan,na,4)+1j*np.random.randn(ndir,nchan,na,4))
        # ANTENNA X DDE X CHAN
        sols_shape = (nchan,na,ndir,4)
        sols=cuda.pagelocked_empty(sols_shape, np.complex128)
        sols[:] = np.ones(sols_shape)+1j*np.zeros(sols_shape)

        # Matrix containing information, here just the reference frequency
        # to estimate the flux from spectral index
        Info=np.array([1e6],np.float64)

#        P1=predict.predictSols(vis, A0, A1, uvw, lms, WaveL, Sols, Info)

        f = open('test.txt','w')

        baselines_per_block = 8 if nbl > 8 else nbl
        ddes_per_block = 32 if ndir > 32 else ndir
        foreground_stream,background_stream = shared_data.stream[0], shared_data.stream[1]

        baseline_blocks = (nbl + baselines_per_block - 1) / baselines_per_block
        dde_blocks = (ndir + ddes_per_block - 1) / ddes_per_block

        print
        print 'block = (' + str(baselines_per_block) + ',' + str(ddes_per_block) + ',1)'
        print 'grid = (' + str(baseline_blocks) + ',' + str(dde_blocks) + ',1)'

#        print 'vis shape', vis[0,:,:,:].shape
#        print 'vis bytes', vis[0,:,:,:].nbytes

        vis_gpu = cuda.mem_alloc(vis[0,:,:,:].nbytes)
        uvw_gpu = cuda.mem_alloc(uvw.nbytes)
        lms_gpu = cuda.mem_alloc(lms.nbytes)
        A0_gpu = cuda.mem_alloc(A0.nbytes)
        A1_gpu = cuda.mem_alloc(A1.nbytes)
        sols_gpu = cuda.mem_alloc(sols[0,:,:,:].nbytes)

        for chan in range(nchan):
            # TODO: Fix the .copy(), its expensive
            cuda.memcpy_htod_async(vis_gpu, vis[chan,:,:,:],
                stream=foreground_stream)
            cuda.memcpy_htod_async(uvw_gpu, uvw,
                stream=foreground_stream)
            cuda.memcpy_htod_async(lms_gpu, lms,
                stream=foreground_stream)
            cuda.memcpy_htod_async(A0_gpu, A0,
                stream=foreground_stream)
            cuda.memcpy_htod_async(A1_gpu, A1,
                stream=foreground_stream)
            # TODO: Fix the .copy(), its expensive
            cuda.memcpy_htod_async(sols_gpu, sols[chan,:,:,:],
                stream=foreground_stream)

            self.kernel(vis_gpu, uvw_gpu, lms_gpu,
                A0_gpu, A1_gpu, sols_gpu, wavelength[chan],
                np.int32(ndir), np.int32(na), np.int32(nbl),
                stream=foreground_stream,
                block=(baselines_per_block,ddes_per_block,1),
                grid=(baseline_blocks,dde_blocks,1))

            foreground_stream,background_stream = background_stream,foreground_stream

            cuda.memcpy_dtoh_async(vis[chan,:,:,:], vis_gpu,
                stream=foreground_stream)
 
            f.write('Channel ' + str(chan) + '\n')
            for v in vis[chan,:,:,:]:
                f.write(str(v) + '\n')

        f.close()
        vis_gpu.free()
        uvw_gpu.free()
        lms_gpu.free()
        A0_gpu.free()
        A1_gpu.free()
        sols_gpu.free()

    def post_execution(self, shared_data):
        pass