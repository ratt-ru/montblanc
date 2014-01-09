import argparse
import os.path
import sys

import numpy as np
import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

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
    def __init__(self):
        super(StreamNode1, self).__init__()
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

class StreamNode2(Node):
    def __init__(self):
        super(StreamNode2, self).__init__()
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

        print 'Initialising pipeline'

        try:
            for node in self.pipeline:
                print '\tInitialising node \'' + node.description() + '\'',
                node.initialise(shared_data)
                print 'Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline initialisation', e
            return shared_data
        except Exception, e:
            print
            print 'Unexpected exception occurred during RIME pipeline initialisation', e
            return shared_data

        print 'Initialisation of pipeline complete'
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
            return shared_data
        except Exception, e:
            print
            print 'Unexpected exception occurred during RIME pipeline execution', e
            return shared_data

        print 'Execution of pipeline complete'
        print 'Shutting down pipeline'

        try:
            for node in self.pipeline:
                print '\tShutting down node \'' + node.description() + '\'',
                node.shutdown(shared_data)
                print 'Done'
        except PipeLineError as e:
            print
            print 'Pipeline Error occurred during RIME pipeline shutdown', e
            return shared_data            
        except Exception, e:
            print
            print 'Unexpected exception occurred during RIME pipeline shutdown', e
            return shared_data            

        print 'Shutdown of pipeline Complete'

        return shared_data

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

    sp = PipedRimes([GPUNode(), GPUNode()])
    data = sp.execute()

    print data.a_cpu[data.N-10:]
    print data.result[data.N-10:]


    sp = PipedRimes([StreamNode1(), StreamNode2(), StreamNode3()])
    data = sp.execute()



    print sp

if __name__ == "__main__":
    sys.exit(main())
