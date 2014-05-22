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