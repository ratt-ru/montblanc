import numpy as np
import types

from weakref import WeakKeyDictionary

import montblanc
import montblanc.util as mbu

from montblanc.solvers import RIMESolver
from montblanc.config import SolverConfig as Options

class CUDAArrayDescriptor(object):
    """ Descriptor class for pycuda.gpuarrays on the GPU """
    def __init__(self, record_key, default=None):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        instance.check_array(self.record_key, value)
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class CUDASolver(RIMESolver):
    """ Solves the RIME using CUDA """
    def __init__(self, slvr_cfg):
        super(CUDASolver, self).__init__(slvr_cfg)

        import pycuda.driver as cuda

        # Store the context, choosing the default if not specified
        ctx = slvr_cfg.get(Options.CONTEXT, None)

        if ctx is None:
            raise Exception(('No CUDA context was supplied'
                ' provided in the slvr_cfg argument of {c}')
                    .format(c=self.__class__.__name__))

        if isinstance(ctx, list):
            ctx = ctx[0]

        # Create a context wrapper
        self.context = mbu.ContextWrapper(ctx)

        # Figure out the integer compute cability of the device
        # associated with the context
        with self.context as ctx:
            cc_tuple = cuda.Context.get_device().compute_capability()
        # np.dot((3,5), (100,10)) = 3*100 + 5*10 = 350 for Kepler
        self.cc = np.int32(np.dot(cc_tuple, (100,10)))

        # Configure our solver pipeline
        pipeline = slvr_cfg.get('pipeline', None)

        if pipeline is None:
            pipeline = montblanc.factory.get_empty_pipeline(slvr_cfg)
        self.pipeline = pipeline

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        A = super(CUDASolver, self).register_array(
            name, shape, dtype, registrant, **kwargs)

        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray

        # Create descriptors on the class instance, even though members
        # may not necessarily be created on object instances. This is so
        # that if someone registers an array but doesn't ask for it to be
        # created, we have control over it, if at some later point they wish
        # to do a
        #
        # slvr.blah = ...
        #

        # TODO, there's probably a better way of figuring out if a descriptor
        # is set on the class
        #if not hasattr(CUDASolver, A.name):
        if A.name not in CUDASolver.__dict__:
            setattr(CUDASolver, A.name, CUDAArrayDescriptor(record_key=A.name))

        # Create an empty array
        cpu_ary = np.empty(shape=A.shape, dtype=A.dtype)                
        data_source =self._slvr_cfg[Options.DATA_SOURCE]

        # If we're creating test data, initialise the array with
        # data from the test key, don't initialise if we've been
        # explicitly told the array should be empty, otherwise
        # set the defaults
        if data_source == Options.DATA_SOURCE_TEST:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_TEST, None))
        elif data_source == Options.DATA_SOURCE_EMPTY:
            pass
        else:
            self.init_array(name, cpu_ary,
                kwargs.get(Options.DATA_SOURCE_DEFAULT, None))               

        # We don't use gpuarray.zeros, since it fails for
        # a zero-length array. This is kind of bad since
        # the gpuarray returned by gpuarray.empty() doesn't
        # have GPU memory allocated to it.
        with self.context as ctx:
            # Query free memory on this context
            (free_mem,total_mem) = cuda.mem_get_info()

            montblanc.log.debug("Allocating GPU memory "
                "of size {s} for array '{n}'. {f} free "
                "{t} total on device.".format(n=name,
                    s=self.fmt_bytes(self.array_bytes(A)),
                    f=self.fmt_bytes(free_mem),
                    t=self.fmt_bytes(total_mem)))

            gpu_ary = gpuarray.empty(shape=A.shape, dtype=A.dtype)

            # If the array length is non-zero initialise it
            if (data_source != Options.DATA_SOURCE_EMPTY and
                np.product(A.shape) > 0):
                gpu_ary.set(cpu_ary)
            
            setattr(self, A.name, gpu_ary)

        # Should we create a setter for this property?
        transfer_method = kwargs.get('transfer_method', True)

        # OK, we got a boolean for the kwarg, create a default transfer method
        if isinstance(transfer_method, types.BooleanType) and transfer_method is True:
            # Create the transfer method
            def transfer(self, npary):
                self.check_array(A.name, npary)
                with self.context:
                    getattr(self,A.name).set(npary)

            transfer_method = types.MethodType(transfer,self)
        # Otherwise, we can just use the supplied kwarg
        elif isinstance(transfer_method, types.MethodType):
            pass
        else:
            raise TypeError(('transfer_method keyword argument set '
                'to an invalid type %s') % (type(transfer_method)))

        # Name the transfer method
        transfer_method_name = self.transfer_method_name(name)
        setattr(self,  transfer_method_name, transfer_method)
        # Create a docstring!
        getattr(transfer_method, '__func__').__doc__ = \
        """
        Transfers the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (A.name,A.name)

        # Should we create a getter for this property?
        retrieve_method = kwargs.get('retrieve_method', True)

        # OK, we got a boolean for the kwarg, create a default retrieve method
        if isinstance(retrieve_method, types.BooleanType) and retrieve_method is True:
            # Create the retrieve method
            def retrieve(self):
                with self.context:
                    return getattr(self,A.name).get()

            retrieve_method = types.MethodType(retrieve,self)
        # Otherwise, we can just use the supplied kwarg
        elif isinstance(retrieve_method, types.MethodType):
            pass
        else:
            raise TypeError(('retrieve_method keyword argument set '
                'to an invalid type %s') % (type(retrieve_method)))

        # Name the retrieve method
        retrieve_method_name = self.retrieve_method_name(name)
        setattr(self,  retrieve_method_name, retrieve_method)
        # Create a docstring!
        getattr(retrieve_method, '__func__').__doc__ = \
        """
        Retrieve the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (A.name,A.name)

    def transfer_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'transfer_' + name

    def retrieve_method_name(self, name):
        """ Constructs a transfer method name, given the array name """
        return 'retrieve_' + name

    def solve(self):
        """ Solve the RIME """
        with self.context as ctx:
            self.pipeline.execute(self)

    def initialise(self):
        with self.context as ctx:
            self.pipeline.initialise(self)

    def shutdown(self):
        """ Stop the RIME solver """
        with self.context as ctx:
            self.pipeline.shutdown(self)        

