import argparse
import os.path
import sys

import numpy as np
import pycuda.autoinit

from node import *
from rime3D import *

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

        self.event_names = [StreamNode1.INIT, \
            StreamNode1.PRE, StreamNode1.EXEC, \
            StreamNode1.POST, StreamNode1.SHUTDOWN]

    def initialise(self, shared_data):
        stream = [cuda.Stream() for k in range(self.K)]
        event = [dict([(en, cuda.Event()) for en in self.event_names]) for k in range(self.K)]

        for k in range(self.K):
            event[k][StreamNode1.INIT].record(stream[k])
            event[k][StreamNode1.PRE].record(stream[k])
            event[k][StreamNode1.EXEC].record(stream[k])
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


    sp = PipedRimes([StreamNode1(), Rime3D(), StreamNode3()])
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
