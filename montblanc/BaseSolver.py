import numpy as np
import sys
import types

from weakref import WeakKeyDictionary

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc
import montblanc.factory
import montblanc.util

class ArrayRecord(object):
    """ Records information about an array """
    def __init__(self, name, sshape, shape, dtype, registrant, has_cpu_ary, has_gpu_ary):
        self.name = name
        self.sshape = sshape
        self.shape = shape
        self.dtype = dtype
        self.registrant = registrant
        self.has_cpu_ary = has_cpu_ary
        self.has_gpu_ary = has_gpu_ary

class PropertyRecord(object):
    """ Records information about a property """
    def __init__(self, name, dtype, default, registrant):
        self.name = name
        self.dtype = dtype
        self.default = default
        self.registrant = registrant

class PipelineDescriptor(object):
    """ Descriptor class for pipelines """
    def __init__(self):
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,None)

    def __set__(self, instance, value):
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class PropertyDescriptor(object):
    """ Descriptor class for properties """
    def __init__(self, record_key, default=None, ):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        dtype = instance.properties[self.record_key].dtype
        self.data[instance] = dtype(value)

    def __delete__(self, instance):
        del self.data[instance]

class CPUArrayDescriptor(object):
    """ Descriptor class for NumPy ndarrays arrays on the CPU """
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

class GPUArrayDescriptor(object):
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

class Parameter(object):
    """ Descriptor class for describing parameters """
    def __init__(self, default=None):
        self.default = default
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Negative parameter value: %s' % value )
        self.data[instance] = value

    def __delete__(self, instance):
        del self.data[instance]

class Solver(object):
    """ Base class for data shared amongst pipeline nodes.

    In practice, nodes will be responsible for creating,
    updating and deleting members of this class.
    Its not a complicated beast.
    """
    pass

DEFAULT_NA=3
DEFAULT_NBL=montblanc.nr_of_baselines(DEFAULT_NA)
DEFAULT_NCHAN=4
DEFAULT_NTIME=10
DEFAULT_NPSRC=2
DEFAULT_NGSRC=1
DEFAULT_NSRC=DEFAULT_NPSRC + DEFAULT_NGSRC
DEFAULT_NVIS=DEFAULT_NBL*DEFAULT_NCHAN*DEFAULT_NTIME
DEFAULT_DTYPE=np.float32

class BaseSolver(Solver):
    """ Class that holds the elements for solving the RIME """
    na = Parameter(DEFAULT_NA)
    nbl = Parameter(DEFAULT_NBL)
    nchan = Parameter(DEFAULT_NCHAN)
    ntime = Parameter(DEFAULT_NTIME)
    npsrc = Parameter(DEFAULT_NPSRC)
    ngsrc = Parameter(DEFAULT_NGSRC)
    nsrc = Parameter(DEFAULT_NSRC)
    nvis = Parameter(DEFAULT_NVIS)

    pipeline = PipelineDescriptor()

    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, dtype=DEFAULT_DTYPE,
        pipeline=None, **kwargs):
        """
        BaseSolver Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
        Keyword Arguments:
            device : pycuda.device.Device
                CUDA device to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays.
            auto_correlations: boolean
                if True, take auto-correlations into account when
                calculating the number of baselines.
        """

        super(BaseSolver, self).__init__()

        autocor = kwargs.get('auto_correlations',True)

        # Configure our problem dimensions. Number of
        # - antenna
        # - baselines
        # - channels
        # - timesteps
        # - point sources
        # - gaussian sources
        self.na = na
        self.nbl = nbl = montblanc.nr_of_baselines(na,autocor)
        self.nchan = nchan
        self.ntime = ntime
        self.npsrc = npsrc
        self.ngsrc = ngsrc
        self.nsrc = nsrc = npsrc + ngsrc
        self.nvis = nbl*nchan*ntime

        if nsrc == 0:
            raise ValueError, ('The number of sources, or, ',
                'the sum of npsrc and ngsrc, must be greater than zero')

        # Configure our floating point and complex types
        if dtype == np.float32:
            self.ct = ct = np.complex64
        elif dtype == np.float64:
            self.ct = ct = np.complex128
        else:
            raise TypeError, ('Must specify either np.float32 ',
                'or np.float64 for dtype')

        self.ft = ft = dtype

        # Store the device, choosing the default if not specified
        self.device = kwargs.get('device')

        if self.device is None:
            import pycuda.autoinit
            self.device = pycuda.autoinit.device

        # Figure out the integer compute cability of the device
        cc_tuple = self.device.compute_capability()
        # np.dot((3,5), (100,10)) = 3*100 + 5*10 = 350 for Kepler
        self.cc = np.int32(np.dot(cc_tuple, (100,10)))

        # Dictionaries to store records about our arrays and properties
        self.arrays = {}
        self.properties = {}

        # Should we store CPU versions of the GPU arrays
        self.store_cpu = kwargs.get('store_cpu', False)

        # Configure our solver pipeline
        if pipeline is None:
            pipeline = montblanc.factory.get_empty_pipeline()
        self.pipeline = pipeline

    def get_actual_dtype(self,sdtype):
        """
        Substitutes string dtype parameters with actual
        NumPy dtypes.

        Parameters
        ----------
            sdtype : string defining the dtype

        """

        return {
            'ft' : self.ft,
            'ct' : self.ct,
            'int' : np.int64
        }[sdtype]

    def get_numeric_shape(self, sshape, ignore=None):
        """
        Substitutes string values in the supplied shape parameter
        with properties registered on this BaseSolver object.

        Parameters
        ----------
            sshape : tuple composed of integers and strings.
                The strings should related to integral properties
                registered with this Solver object
            ignore : list
                A list of tuple strings to ignore

        >>> print self.get_numeric_shape((4,'na','ntime'),ignore=['ntime'])
        (4, 3)
        """
        D = self.get_properties()
        if ignore is None: ignore = []
        
        if type(sshape) is not tuple:
            raise TypeError, 'shape argument must be a tuple'

        if type(ignore) is not list:
            raise TypeError, 'ignore argument must be a list'

        def tup_replace(value):
            """ Replace strings in a tuple with the supplied dictionary value """
            try:
                replace_value = D[value]
            except KeyError:
                if not np.issubdtype(type(value), np.integer):
                    raise KeyError, ('Unable to replace %s in shape %s ',
                        'with a suitable integral value.') % \
                        (value, sshape, )
                replace_value = value

            return replace_value

        return tuple([tup_replace(v) for v in sshape if v not in ignore])

    def viable_timesteps(self, bytes_available):
        """
        Returns the number of timesteps possible, given the registered arrays
        and a memory budget defined by bytes_available
        """

        # Figure out which arrays have an ntime dimension
        has_time = np.array([ \
            t.sshape.count('ntime') > 0 for t in self.arrays.values()])

        # Get the shape product of each array, EXCLUDING any ntime dimension,
        # multiplied by the size of the array type in bytes.
        products = np.array([ \
            np.product(self.get_numeric_shape(t.sshape, ignore=['ntime'])) * \
            np.dtype(t.dtype).itemsize \
            for t in self.arrays.values()])

        # Determine a linear expression for the bytes
        # required which varies by timestep. y = a + b*x
        a = np.sum(np.logical_not(has_time)*products)
        b = np.sum(has_time*products)

        # Check that if we substitute ntime for x, we agree on the
        # memory requirements
        assert a + b*self.ntime == self.bytes_required()

        # Given the number of bytes available,
        # how many timesteps can we fit in our budget?
        return (bytes_available - a + b - 1) // b

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([montblanc.util.array_bytes(a.shape,a.dtype) 
            for a in self.arrays.itervalues()])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return montblanc.util.fmt_bytes(self.bytes_required())

    def check_array(self, record_key, ary):
        """
        Check that the shape and type of the supplied array matches
        our supplied record
        """
        record = self.arrays[record_key]

        if record.shape != ary.shape:
            raise ValueError, \
                '%s\'s shape %s is different from the shape %s of the supplied argument.' \
                % (record.name, record.shape, ary.shape)

        if record.dtype != ary.dtype:
            raise TypeError, \
                '%s\'s type \'%s\' is different from the type \'%s\' of the supplied argument.' % \
                    (record.name,
                    np.dtype(record.dtype).name,
                    np.dtype(ary.dtype).name)          

    def handle_existing_array(self, old, new, **kwargs):
        """
        Compares old array record against new. Complains
        if there's a mismatch in shape or type. 
        """ 
        should_replace = kwargs.get('replace',False)

        # There's no existing record, or we've been told to replace it
        if old is None or should_replace is True:
            return

        # Check that the shapes are the same
        if old.shape != new.shape:
            raise ValueError, ('\'%s\' array is already registered by '
                '\'%s\' with shape %s different to the supplied %s.') % \
                (old.name,
                old.registrant,
                old.shape,
                new.shape,)

        # Check that the types are the same
        if old.dtype != new.dtype:
            raise ValueError, ('\'%s\' array is already registered by '
                '\'%s\' with type %s different to the supplied %s.') % \
                    (old.name, old.registrant,
                    np.dtype(old.dtype).name,
                    np.dtype(new.dtype).name,)

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        """
        Register an array with this Solver object.

        Parameters
        ----------
            name : string
                name of the array.
            shape : integer/string or tuple of integers/strings
                Shape of the array.
            dtype : data-type
                The data-type for the array.
            registrant : string
                Name of the entity registering this array.
        
        Keyword Arguments
        -----------------
            cpu : boolean
                True if a ndarray called 'name_cpu' should be
                created on the Solver object.
            gpu : boolean
                True if a gpuarray called 'name_gpu' should be
                created on the Solver object.
            shape_member : boolean
                True if a member called 'name_shape' should be
                created on the Solver object.
            dtype_member : boolean
                True if a member called 'name_dtype' should be
                created on the Solver object.
            replace : boolean
                True if existing arrays should be replaced.
        """
        # Try and find an existing version of this array
        old = self.arrays.get(name, None)

        # Should we create arrays?
        want_cpu_ary = kwargs.get('cpu', False) or self.store_cpu
        want_gpu_ary = kwargs.get('gpu', True)

        # Have we already created arrays?
        cpu_ary_exists = True if old is not None and old.has_cpu_ary else False
        gpu_ary_exists = True if old is not None and old.has_gpu_ary else False

        # Determine whether the new record we're creating will have
        # CPU or GPU arrays
        has_cpu_ary = cpu_ary_exists or want_cpu_ary
        has_gpu_ary = gpu_ary_exists or want_gpu_ary

        # Determine whether we need to create cpu/gpu arrays
        create_cpu_ary = not cpu_ary_exists and want_cpu_ary
        create_gpu_ary = not gpu_ary_exists and want_gpu_ary

        # Figure out the actual integral shape
        sshape = shape
        shape = self.get_numeric_shape(sshape)

        if type(dtype) == str:
            dtype = self.get_actual_dtype(dtype)

        # Create a new record
        new = ArrayRecord(
            name=name,
            sshape=sshape,
            shape=shape,
            dtype=dtype,
            registrant=registrant,
            has_cpu_ary=has_cpu_ary,
            has_gpu_ary=has_gpu_ary)

        # Check if the array has been registered previously
        # and if we're allowed to replace it
        self.handle_existing_array(old, new, **kwargs)

        # OK, create/replace a record for this array
        self.arrays[name] = new

        # Attribute names
        cpu_name = montblanc.util.cpu_name(name)
        gpu_name = montblanc.util.gpu_name(name)

        # Create descriptors on the class instance, even though members
        # may not necessarily be created on object instances. This is so
        # that if someone registers an array but doesn't ask for it to be
        # created, we have control over it, if at some later point they wish
        # to do a
        #
        # slvr.blah_cpu = ...
        #

        # TODO, there's probably a better way of figuring out if a descriptor
        # is set on the class
        #if not hasattr(BaseSolver, cpu_name):
        if not BaseSolver.__dict__.has_key(cpu_name):
            setattr(BaseSolver, cpu_name, CPUArrayDescriptor(record_key=name))

        #if not hasattr(BaseSolver, gpu_name):
        if not BaseSolver.__dict__.has_key(gpu_name):
            setattr(BaseSolver, gpu_name, GPUArrayDescriptor(record_key=name))

        # Create an empty cpu array if it doesn't exist
        # and set it on the object instance
        if create_cpu_ary:
            cpu_ary = np.empty(shape=shape, dtype=dtype)
            setattr(self, cpu_name, cpu_ary)

        # Create an empty gpu array if it doesn't exist
        # and set it on the object instance
        # Also create a transfer method for tranferring data to the GPU
        if create_gpu_ary:
            # We don't use gpuarray.zeros, since it fails for
            # a zero-length array. This is kind of bad since
            # the gpuarray returned by gpuarray.empty() doesn't
            # have GPU memory allocated to it.
            gpu_ary = gpuarray.empty(shape=shape, dtype=dtype)

            # Zero the array, if it has non-zero length
            if np.product(shape) > 0: gpu_ary.fill(dtype(0))
            setattr(self, gpu_name, gpu_ary)

        # Create the transfer method
        def transfer(self, npary):
            self.check_array(name, npary)
            if create_cpu_ary: setattr(self,cpu_name,npary)
            if create_gpu_ary: getattr(self,gpu_name).set(npary)

        # Create the method on ourself
        transfer_method_name = montblanc.util.transfer_method_name(name)
        transfer_method = types.MethodType(transfer,self)
        setattr(self,  transfer_method_name, transfer_method)
        # Create a docstring!
        getattr(transfer_method, '__func__').__doc__ = \
        """
        Transfers the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (gpu_name,gpu_name)


        # Set up a member describing the shape
        if kwargs.get('shape_member', False) is True:
            shape_name = montblanc.util.shape_name(name)
            setattr(self, shape_name, shape)

        # Set up a member describing the dtype
        if kwargs.get('dtype_member', False) is True:
            dtype_name = montblanc.util.dtype_name(name)
            setattr(self, dtype_name, dtype)

    def register_arrays(self, array_dicts):
        """
        Register arrays using a dictionary defining the arrays.

        The dictionary should itself contain dictionaries. i.e.

        >>> D = {
            'uvw' : { 'name':'uvw', 'shape':(3,'ntime','nbl'),'dtype':np.float32 },
            'lm' : { 'name':'lm', 'shape':(2,'nsrc'),'dtype':np.float32 }
        }
        """
        for name, ary in array_dicts.iteritems():
            self.register_array(**ary)

    def register_property(self, name, dtype, default, registrant, **kwargs):
        """
        Registers a property with this Solver object

        Parameters
        ----------
            name : string
                The name of the property.
            dtype : data-type
                The data-type of this property
            default : 
                Default value for the property.
            registrant : string
                Name of the entity registering the property.

        Keyword Arguments
        -----------------
            setter : boolean or function
                if True, a default 'set_name' member is created, otherwise not.
                If a method, this is used instead.
            setter_docstring : string
                docstring for the default setter.

        """

        if type(dtype) == str:
            dtype = self.get_actual_dtype(dtype)

        self.properties[name] = pr = PropertyRecord(
            name, dtype, default, registrant)

        # Create the descriptor for this property on the class instance
        setattr(BaseSolver, name, PropertyDescriptor(record_key=name, default=default))
        # Set the descriptor on this object instance
        setattr(self, name, default)

        # Should we create a setter for this property?
        setter = kwargs.get('setter', True)
        setter_name = montblanc.util.setter_name(name)

        # Yes, create a default setter
        if isinstance(setter, types.BooleanType) and setter is True:
            def set(self, value):
                setattr(self,name,value)

            setter_method = types.MethodType(set, self)
            setattr(self, setter_name, setter_method)

            # Set up the docstring, using the supplied one
            # if it is present, otherwise generating a default
            setter_docstring = kwargs.get('setter_docstring', None)
            getattr(setter_method, '__func__').__doc__ = \
                """ Sets property %s to value. """ % (name) \
                if setter_docstring is None else setter_docstring

        elif isinstance(setter, types.MethodType):
            settattr(self, setter_name, setter)
        else:
            raise TypeError, ('setter keyword argument set',
                ' to an invalid type %s' % (type(setter)))

    def register_properties(self, property_dicts):
        """
        Register properties using a dictionary defining the properties.

        The dictionary should itself contain dictionaries. i.e.

        >>> D = {
            'ref_wave' : { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'solver' },
        }
        """
        for name, prop in property_dicts.iteritems():
            self.register_property(**prop)

    def get_array_record(self, name):
        return self.arrays[name]

    def get_properties(self):
        """
        Returns a dictionary of properties related to this Solver object.

        Used in templated GPU kernels.
        """
        slvr = self
        
        D = {
            'na' : slvr.na,
            'nbl' : slvr.nbl,
            'nchan' : slvr.nchan,
            'ntime' : slvr.ntime,
            'npsrc' : slvr.npsrc,
            'ngsrc' : slvr.ngsrc,
            'nsrc'  : slvr.nsrc,
            'nvis' : slvr.nvis,
        }

        for p in self.properties.itervalues():
            D[p.name] = getattr(self,p.name)

        return D

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def gen_array_descriptions(self):
        """ Generator generating strings describing each registered array """
        yield 'Registered Arrays'
        yield '-'*80
        yield montblanc.util.fmt_array_line('Array Name','Size','Type','CPU','GPU','Shape')
        yield '-'*80

        for a in self.arrays.itervalues():
            yield montblanc.util.fmt_array_line(a.name,
                montblanc.util.fmt_bytes(montblanc.util.array_bytes(a.shape, a.dtype)),
                np.dtype(a.dtype).name,
                'Y' if a.has_cpu_ary else 'N',
                'Y' if a.has_gpu_ary else 'N',
                a.sshape)

    def gen_property_descriptions(self):
        """ Generator generating string describing each registered property """
        yield 'Registered Properties'
        yield '-'*80
        yield montblanc.util.fmt_property_line('Property Name',
            'Type', 'Value', 'Default Value')
        yield '-'*80

        for p in self.properties.itervalues():
            yield montblanc.util.fmt_property_line(
                p.name, np.dtype(p.dtype).name,
                getattr(self, p.name), p.default)

    def solve(self):
        """ Solve the RIME """        
        self.pipeline.execute(self)

    def initialise(self):
        self.pipeline.initialise(self)

    def shutdown(self):
        """ Stop the RIME solver """
        self.pipeline.shutdown(self)

    def __enter__(self):
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __str__(self):
        """ Outputs a string representation of this object """
        n_cpu_bytes = np.sum([montblanc.util.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues() if a.has_cpu_ary is True])

        n_gpu_bytes = np.sum([montblanc.util.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues() if a.has_gpu_ary is True])

        w = 20

        l = ['',
            'RIME Dimensions',
            '-'*80,
            '%-*s: %s' % (w,'Antenna', self.na),
            '%-*s: %s' % (w,'Baselines', self.nbl),
            '%-*s: %s' % (w,'Channels', self.nchan),
            '%-*s: %s' % (w,'Timesteps', self.ntime),
            '%-*s: %s' % (w,'Point Sources', self.npsrc),
            '%-*s: %s' % (w,'Gaussian Sources', self.ngsrc),
            '%-*s: %s' % (w,'CPU Memory', montblanc.util.fmt_bytes(n_cpu_bytes)),
            '%-*s: %s' % (w,'GPU Memory', montblanc.util.fmt_bytes(n_gpu_bytes))]

        l.extend([''])
        l.extend([s for s in self.gen_array_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)