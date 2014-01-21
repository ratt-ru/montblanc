import argparse
import os.path
import sys

import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel

class ArrayData(object):
    """ Unused Descriptor Class. For gpuarrays """
    def __init__(self, value=None):
        if value is None:
            value = []

        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = value

class PipeLineError(Exception):
    """ Pipeline Exception base class """
    pass

class SharedData(object):
    """ Base class for data shared amongst pipeline nodes.

    In practice, nodes will be responsible for creating,
    updating and deleting members of this class.
    Its not a complicated beast.
    """
    pass

class Node(object):
    """
    Abstract pipeline node class.

    This class should be extended to provide a concrete implementation of a pipeline node class.

    >>> class GPUNode(Node):
    >>>     def __init__(self):
    >>>         super(type(self), self).__init__()
    >>>     def pre_execution(self, shared_data):
    >>>         shared_data.N = 1024
    >>>         shared_data.a_cpu = np.random.random(shared_data.N)
    >>>         shared_data.a_gpu = gpuarray.to_gpu(shared_data.a_cpu)
    >>>     def post_execution(self, shared_data):
    >>>         del shared.data.N
    >>>         del shared_data.a_gpu
    >>>         del shared_data.a_cpu
    >>>     def execute(self, shared_data):
    >>>         shared_data.result = (shared_data.a_gpu*2./3.).get()
    """

    def __init__(self):
        pass

    def description(self):
        """ Returns a string description of the node """
        #raise NotImplementedError, self.not_implemented_string(type(self).description.__name__)
        return self.__class__.__name__

    def initialise(self, shared_data):
        """
        Abstract method that should be overriden by derived classes.
        This method is called before the execution of the pipeline. Any initialisation
        required by the node should be performed here.

        Keyword arguments:
        shared_data -- object containing data shared amongst pipeline components
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).initialise.__name__)

    def shutdown(self, shared_data):
        """
        Abstract method that should be overriden by derived classes.
        This method is called after the execution of the pipeline. Any cleanup required
        by the node should be performed here.

        Keyword arguments:
        shared_data -- object containing data shared amongst pipeline components
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).shutdown.__name__)

    def execute(self, shared_data):
        """
        Abstract method that should be overriden by derived classes.
        The derived class should implement the code that the node should
        execute. Most likely this will be a PyCUDA GPU kernel of some sort,
        but the pipeline model is general enough to support CPU calls etc.
        
        Keyword arguments:
        shared_data -- object containing data shared amongst pipeline components
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).execute.__name__)

    def pre_execution(self, shared_data):
        """
        Hook for writing code that will be executed prior
        to the execute() method.


        Keyword arguments:
        shared_data -- object containing data shared amongst pipeline components
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).pre_execution.__name__)

    def post_execution(self, shared_data):
        """
        Hook for writing code that will be executed after
        the execute() method.

        Keyword arguments:
        shared_data -- object containing data shared amongst pipeline components
        """
        raise NotImplementedError, self.__not_implemented_string(type(self).post_execution.__name__)

    def __not_implemented_string(self, function_name):
        return ' '.join(['method', function_name, 'not implemented in class',type(self).__name__,'derived from abstract class', Node.__name__])
               
class NullNode(Node):
    def __init__(self): super(NullNode, self).__init__()
    def initialise(self, shared_data): pass
    def shutdown(self, shared_data): pass
    def pre_execution(self, shared_data): pass
    def execute(self, shared_data): pass
    def post_execution(self, shared_data): pass

class GPUNode(NullNode):
    def __init__(self):
        super(GPUNode, self).__init__()
    def pre_execution(self, shared_data):
        shared_data.N = 1024
        shared_data.a_cpu = np.random.random(shared_data.N)
        shared_data.a_gpu = gpuarray.to_gpu(shared_data.a_cpu)
    def execute(self, shared_data):
        shared_data.result = (shared_data.a_gpu*2./3.).get()
    def post_execution(self, shared_data):
        #del shared_data.N
        #del shared_data.a_gpu
        #del shared_data.a_cpu
        pass

class StreamNode1(Node):
    INIT = 'init'
    PRE = 'pre'
    EXEC = 'exec'
    POST = 'post'
    SHUTDOWN = 'shutdown'

    def __init__(self,K=4):
        super(StreamNode1, self).__init__()
        self.K = 4
        self.N = 1024
        self.kernel = ElementwiseKernel(
            "float * in, float a, float * out",
#            "out[i] = a*in[i]",
            "out[i]=a*in[i]",
            "timestwo")
        self.event_names = [StreamNode1.INIT, \
            StreamNode1.PRE, StreamNode1.EXEC, StreamNode1.POST, StreamNode1.SHUTDOWN]

    def initialise(self, shared_data):
        stream = [cuda.Stream() for k in range(self.K)]
        event = [dict([(en, cuda.Event()) for en in self.event_names]) for k in range(self.K)]

        a_cpu = np.random.random(self.N*self.N).astype(np.float32)

        for k in range(self.K):
            event[k][StreamNode1.INIT].record(stream[k])

        a_gpu = gpuarray.to_gpu_async(a_cpu, stream=stream[0])
        b_gpu = gpuarray.zeros_like(a_gpu)

        for k in range(self.K):
            event[k][StreamNode1.PRE].record(stream[k])

#        print
#        print 'b_gpu', b_gpu.get_async(stream=stream[0])[:10]

        self.kernel(a_gpu, 2., b_gpu, stream=stream[0])
        b_cpu = b_gpu.get_async(stream=stream[0])

#        assert (b_cpu*2 == a_cpu).all()

#        print 'a_cpu', a_cpu[:10]
#        print 'a_gpu', a_gpu[:10]
#        print 'b_cpu', b_cpu[:10]

        for k in range(self.K):
            event[k][StreamNode1.EXEC].record(stream[k])

        a_gpu = gpuarray.to_gpu_async(a_cpu, stream=stream[0])

        for k in range(self.K):
            event[k][StreamNode1.POST].record(stream[k])

        shared_data.stream = stream
        shared_data.event = event

    def shutdown(self, shared_data):
        for k in range(self.K):
            shared_data.event[k][StreamNode1.SHUTDOWN].record(shared_data.stream[k])

#        for k in range(self.K):
        if True:
            k = 0
            event_streams = shared_data.event[k]
            init_event = event_streams[StreamNode1.INIT]
            pre_event = event_streams[StreamNode1.PRE]
            exec_event = event_streams[StreamNode1.EXEC]
            post_event = event_streams[StreamNode1.POST]
            shutdown_event = event_streams[StreamNode1.SHUTDOWN]

            print
            print StreamNode1.INIT, init_event.time_till(init_event), 'ms'
            print StreamNode1.PRE, pre_event.time_till(pre_event), 'ms'
            print StreamNode1.EXEC, pre_event.time_till(exec_event), 'ms'
            print StreamNode1.POST, pre_event.time_till(post_event), 'ms'
            print StreamNode1.SHUTDOWN, pre_event.time_till(shutdown_event), 'ms'


        del shared_data.stream
        del shared_data.event
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        pass
    def post_execution(self, shared_data):
        pass

class StreamNode2(Node):
    def __init__(self):
        super(StreamNode2, self).__init__()
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

class StreamNodeOld2(Node):
    def __init__(self):
        super(StreamNode2, self).__init__()
    def initialise(self, shared_data):
        self.N = 1024
        self.a_cpu = np.random.random(self.N).astype(np.float64)
        self.a_gpu = gpuarray.to_gpu_async(self.a_cpu, stream=shared_data.stream[0])
        self.mod = SourceModule("""
__global__ void my_kernel(double *d)
{
    const int i = threadIdx.x;
    d[i] = d[i] * 2.0;
}
""")
        self.kernel = self.mod.get_function('my_kernel')

    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        self.kernel(self.a_gpu, block=(self.N,1,1), stream=shared_data.stream[0])
        array = self.a_gpu.get_async(stream=shared_data.stream[0])
        assert np.allclose(array,self.a_cpu*2.).all(), 'May fail for > 512'
    def post_execution(self, shared_data):
        pass                

class StreamNode3(Node):
    def __init__(self):
        super(StreamNode3, self).__init__()
    def initialise(self, shared_data):
        pass
    def shutdown(self, shared_data):
        pass
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        pass
    def post_execution(self, shared_data):
        pass                

class PipedRimes:
    """ Class describing a pipeline of RIME equations """
    def __init__(self, node_list=None):
        """ Initialise the pipeline with a list of nodes

        Keyword arguments:
            node_list -- A list of nodes defining the pipeline

        >>> pipeline = PipeRimes([InitNode(), \\
            ProcessingNode1(), ProcessingNode2(), CleanupNode()])
        """
        if node_list is None: node_list = [NullNode()]

        if type(node_list) != list:
            raise ValueError, 'node_list argument is not a list'

        self.pipeline = node_list

    def execute(self):
        """ Iterates over the pipeline of nodes, executing the functionality contained
        in each.

        Returns a shared data object that is passed amongst the pipeline
        components. The intention here is that components will create some
        sort of result member variable that is accessible after the pipeline
        execution is complete.

        >>> pipeline = PipeRimes([InitNode(), \\
            ProcessingNode1(), ProcessingNode2(), CleanupNode()])
        >>> data = pipe.execute()
        >>> print data.result # result member created by one of the nodes
        """
        shared_data = SharedData()

        if self.__init_pipeline(shared_data) is True:
            self.__execute_pipeline(shared_data)
        self.__shutdown_pipeline(shared_data)

        return shared_data

    def __init_pipeline(self, shared_data):
        print 'Initialising pipeline'

        try:
            for node in self.pipeline:
                print '\tInitialising node \'' + node.description() + '\'',
                node.initialise(shared_data)
                print 'Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline initialisation', e
            return False
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred duri8\ng RIME pipeline initialisation', e
#            return False

        print 'Initialisation of pipeline complete'
        return True

    def __execute_pipeline(self, shared_data):
        print 'Executing pipeline'

        try:
            for node in self.pipeline:
                print '\tExecuting node \'' + node.description() + '\'',
                node.pre_execution(shared_data)
                print 'pre',
                node.execute(shared_data)
                print 'execute',
                node.post_execution(shared_data)
                print 'post Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline execution', e
            return False
#        except Exception, e:
#            print
#            print 'Unexpected exception occurred during RIME pipeline execution', e
#            return False

        print 'Execution of pipeline complete'
        return False

    def __shutdown_pipeline(self, shared_data):
        print 'Shutting down pipeline'

        success = True

        # Even if shutdown fails, keep trying on the other nodes
        for node in self.pipeline:
            try:
                    print '\tShutting down node \'' + node.description() + '\'',
                    node.shutdown(shared_data)
                    print 'Done'
            except PipeLineError as e:
                print
                print 'Pipeline Error occurred during RIME pipeline shutdown', e
                success = False
#            except Exception, e:
#                print
#                print 'Unexpected exception occurred during RIME pipeline shutdown', e
#                success = False

        print 'Shutdown of pipeline Complete'
        return success

    def __str__(self):
    	return ' '.join([node.description() for node in self.pipeline])

def is_valid_file(parser, arg):
    """

    'Checks that the supplied file exists'

    """
    if not os.path.exists(arg):
        parser.error("The file %s does not exist." % arg)
    else:
        return open(arg, 'r')

def main(argv=None):
    """
    'Main entry point for the RIME Pipeline script'
    """
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description='RIME Pipeline Prototyping')
    parser.add_argument('-i','--input-image', dest="inputfile",
        help='Input Image', required=False, metavar="FILE",
        type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-s','--scale', help='Scale', type=float, default=10.)
    parser.add_argument('-d','--depth', help='Quadtree Depth', type=int, default=8)
    parser.add_argument('-g','--image-depth',  dest='imagedepth', help='Image Depth', type=int, default=8)
    args = parser.parse_args(argv[1:])

#    sp = PipedRimes([GPUNode(), GPUNode()])
#    data = sp.execute()

#    print data.a_cpu[data.N-10:]
#    print data.result[data.N-10:]


    sp = PipedRimes([StreamNode1(), StreamNode2(), StreamNode3()])
    data = sp.execute()



    print sp

if __name__ == "__main__":
    sys.exit(main())


serial_predict = self.mod = SourceModule("""
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
    pycuda::complex<double> * solution,
    int ndir, int nchan, int na, int nrows)
{
//    const int i = blockIdx.x*blockDim.x + threadIdx.x;

    const pycuda::complex<double> I = pycuda::complex<double>(0,1);
    const pycuda::complex<double> c0 = 2.0*CUDART_PI_F*I;

    const double refwave = 1e6;

    pycuda::complex<double> * p0 = VisIn;
    double * p1 = UVWin;

    return;

    for(int dd=0;dd<ndir;dd++)
    {
        double l=LM[0];
        double m=LM[1];
        double fI=LM[2]; // flux
        double alpha=LM[3];
        double fQ=LM[4];
        double fU=LM[5];
        double fV=LM[6];        
        double n = sqrt(1.-l*l-m*m)-1.;

        LM+=6;
        VisIn = p0;
        UVWin = p1;

        pycuda::complex<double> sky[4] =
        {
            fI+fQ,
            fU+I*fV,
            fU-I*fV,
            fI-fQ
        };

        double phase = UVWin[0]*l + UVWin[1]*m + UVWin[2]*n;
        UVWin += 3;
    #if 1

        for(int bl=0; bl<nrows; ++bl)
        {
            for(int ch=0; ch<nchan; ++ch)
            { 
                pycuda::complex<double> c1=c0/wavelength[ch];
                double this_flux=pow(refwave/wavelength[ch],alpha);
                pycuda::complex<double> result=this_flux*pycuda::exp(phase*c1);

                pycuda::complex<double> JJ[4];
                //Sols shape: Nd, Nf, Na, 4
                pycuda::complex<double> * J0
                    = solution + dd*na*nchan*4 + ch*na*4 + A0[bl]*4;
                pycuda::complex<double> * J1
                    = solution + dd*na*nchan*4 + ch*na*4 + A1[bl]*4;

                Product2by2(J0, J1, JJ);

                VisIn[0] += JJ[0]*result;
                VisIn[1] += JJ[1]*result;
                VisIn[2] += JJ[2]*result;
                VisIn[3] += JJ[3]*result;

                VisIn += 4;
            }
        }

    #endif
    }    
}
""")
