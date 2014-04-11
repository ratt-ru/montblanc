import argparse
import os.path
import sys

import numpy as np
import pycuda.autoinit

from node import *
from RimeBK import *
from RimeEBK import *
from RimeMultiply import *
from RimeJonesReduce import *
from TestSharedData import TestSharedData

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

    def execute(self, shared_data=None):
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

        if shared_data is None:
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

    # Set up various thing that aren't possible in PyCUDA
    crimes.setup_cuda()

    sp = PipedRimes([RimeEBK(), RimeJonesReduce()])

    sd = TestSharedData(na=7,nchan=32,ntime=10,nsrc=200)
    print 'Using', sd.gpu_mem()/(1024*1024), 'MB of GPU memory.'

    sp.execute(sd)

    print sp

if __name__ == "__main__":
    sys.exit(main())

