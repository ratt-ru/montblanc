import argparse
import os.path
import sys

import numpy as np
import pycuda.autoinit

from node import *
from rime3D import *
from rime2D import *
from RimeJonesBK import *
from RimeJonesMultiply import *

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

class StartNode(Node):
    INIT = 'init'
    PRE = 'pre'
    EXEC = 'exec'
    POST = 'post'
    SHUTDOWN = 'shutdown'

    def __init__(self,K=4):
        super(StartNode, self).__init__()
        self.K = 4

        self.event_names = [StartNode.INIT, \
            StartNode.PRE, StartNode.EXEC, \
            StartNode.POST, StartNode.SHUTDOWN]

    def initialise(self, shared_data):
        stream = [cuda.Stream() for k in range(self.K)]
        event = [dict([(en, cuda.Event()) for en in self.event_names]) for k in range(self.K)]

        for k in range(self.K):
            event[k][StartNode.INIT].record(stream[k])
            event[k][StartNode.PRE].record(stream[k])
            event[k][StartNode.EXEC].record(stream[k])
            event[k][StartNode.POST].record(stream[k])

        shared_data.stream = stream
        shared_data.event = event

    def shutdown(self, shared_data):
        for k in range(self.K):
            shared_data.event[k][StartNode.SHUTDOWN].record(shared_data.stream[k])

#        for k in range(self.K):
        if True:
            k = 0
            event_streams = shared_data.event[k]
            init_event = event_streams[StartNode.INIT]
            pre_event = event_streams[StartNode.PRE]
            exec_event = event_streams[StartNode.EXEC]
            post_event = event_streams[StartNode.POST]
            shutdown_event = event_streams[StartNode.SHUTDOWN]

            print
            print StartNode.INIT, init_event.time_till(init_event), 'ms'
            print StartNode.PRE, pre_event.time_till(pre_event), 'ms'
            print StartNode.EXEC, pre_event.time_till(exec_event), 'ms'
            print StartNode.POST, pre_event.time_till(post_event), 'ms'
            print StartNode.SHUTDOWN, pre_event.time_till(shutdown_event), 'ms'

        del shared_data.stream
        del shared_data.event
    def pre_execution(self, shared_data):
        pass
    def execute(self, shared_data):
        pass
    def post_execution(self, shared_data):
        pass

class FinalNode(Node):
    def __init__(self):
        super(FinalNode, self).__init__()
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

    sp = PipedRimes([StartNode(), RimeJonesBK(), RimeJonesMultiply(), FinalNode()])
    data = sp.execute()

    print sp

if __name__ == "__main__":
    sys.exit(main())

