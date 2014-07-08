import numpy as np
import sys
import types

from weakref import WeakKeyDictionary

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc

class ArrayRecord(object):
    """ Records information about an array """
    def __init__(self, name, shape, dtype, registrant, has_cpu_ary, has_gpu_ary):
        self.name = name
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

def get_nr_of_baselines(na):
    """ Compute the number of baselines for the 
    given number of antenna """
    return (na*(na-1))//2

class SharedData(object):
    """ Base class for data shared amongst pipeline nodes.

    In practice, nodes will be responsible for creating,
    updating and deleting members of this class.
    Its not a complicated beast.
    """
    pass

DEFAULT_NA=3
DEFAULT_NBL=get_nr_of_baselines(DEFAULT_NA)
DEFAULT_NCHAN=4
DEFAULT_NTIME=10
DEFAULT_NPSRC=2
DEFAULT_NGSRC=1
DEFAULT_NSRC=DEFAULT_NPSRC + DEFAULT_NGSRC
DEFAULT_NVIS=DEFAULT_NBL*DEFAULT_NCHAN*DEFAULT_NTIME
DEFAULT_DTYPE=np.float32

class BaseSharedData(SharedData):
    """ Class defining the RIME Simulation Parameters. """
    na = Parameter(DEFAULT_NA)
    nbl = Parameter(DEFAULT_NBL)
    nchan = Parameter(DEFAULT_NCHAN)
    ntime = Parameter(DEFAULT_NTIME)
    npsrc = Parameter(DEFAULT_NPSRC)
    ngsrc = Parameter(DEFAULT_NGSRC)
    nsrc = Parameter(DEFAULT_NSRC)
    nvis = Parameter(DEFAULT_NVIS)

    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, dtype=DEFAULT_DTYPE, **kwargs):
        """
        BaseSharedData Constructor

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
                if True, store cpu versions of the kernel arrays
                within the GPUSharedData object.
        """

        super(BaseSharedData, self).__init__()

        # Configure our problem dimensions. Number of
        # - antenna
        # - baselines
        # - channels
        # - timesteps
        # - point sources
        # - gaussian sources
        self.na = na
        self.nbl = nbl = get_nr_of_baselines(na)
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

    @staticmethod
    def __cpu_name(name):
        """ Constructs a name for the CPU version of the array """
        return name + '_cpu'

    @staticmethod
    def __gpu_name(name):
        """ Constructs a name for the GPU version of the array """
        return name + '_gpu'

    @staticmethod
    def __transfer_method_name(name):
        """ Constructs a transfer method name, given the array name """
        return 'transfer_' + name

    @staticmethod
    def __shape_name(name):
        """ Constructs a name for the array shape member, based on the array name """
        return name + '_shape'

    @staticmethod
    def __dtype_name(name):
        """ Constructs a name for the array data-type member, based on the array name """
        return name + '_dtype'

    @staticmethod
    def __setter_name(name):
        """ Constructs a name for the property, based on the property name """
        return 'set_' + name

    @staticmethod
    def fmt_array_line(name,size,dtype,cpu,gpu,shape):
        """ Format array parameters on an 80 character width line """
        return '%-*s%-*s%-*s%-*s%-*s%-*s' % (
            20,name,
            10,size,
            15,dtype,
            4,cpu,
            4,gpu,
            20,shape)

    @staticmethod
    def fmt_property_line(name,dtype,value,default):
        return '%-*s%-*s%-*s%-*s' % (
            20,name,
            10,dtype,
            20,value,
            20,default)

    @staticmethod
    def fmt_bytes(nbytes):
        """ Returns a human readable string, given the number of bytes """
        for x in ['B','KB','MB','GB']:
            if nbytes < 1024.0:
                return "%3.1f%s" % (nbytes, x)
            nbytes /= 1024.0
        
        return "%.1f%s" % (nbytes, 'TB')

    @staticmethod
    def array_bytes(shape, dtype):
        """ Estimates the memory in bytes required for an array of the supplied shape and dtype """
        return np.product(shape)*np.dtype(dtype).itemsize

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([BaseSharedData.array_bytes(a.shape,a.dtype) 
            for a in self.arrays.itervalues()])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return BaseSharedData.fmt_bytes(self.bytes_required())

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
            raise Warning, ('\'%s\' array ws already registered by '
                '\'%s\ with shape %s different to the supplied %s.') % \
                (old.name,
                old.registrant,
                old.shape,
                new.shape,)

        # Check that the types are the same
        if old.dtype != new.dtype:
            raise Warning, ('\'%s\' array is already registered by '
                '\'%s\' with type %s different to the supplied %s.') % \
                    (old.name, old.registrant,
                    np.dtype(old.dtype).name,
                    np.dtype(new.dtype).name,)

    def register_array(self, name, shape, dtype, registrant, **kwargs):
        """
        Register an array with this SharedData object.

        Parameters
        ----------
            name : string
                name of the array.
            shape : integer or tuple of integers
                Shape of the array.
            dtype : data-type
                The data-type for the array.
            registrant : string
                Name of the entity registering this array.
        
        Keyword Arguments
        -----------------
            cpu : boolean
                True if a ndarray called 'name_cpu' should be
                created on the SharedData object.
            gpu : boolean
                True if a gpuarray called 'name_gpu' should be
                created on the SharedData object.
            shape_member : boolean
                True if a member called 'name_shape' should be
                created on the SharedData object.
            dtype_member : boolean
                True if a member called 'name_dtype' should be
                created on the SharedData object.
            replace : boolean
                True if existing arrays should be replaced.
        """
        # Try and find an existing version of this array
        old = self.arrays.get(name, None)

        has_cpu_ary = kwargs.get('cpu', False)
        has_gpu_ary = kwargs.get('gpu', True)

        # Assume that we don't have any arrays yet
        cpu_ary_exists = gpu_ary_exists = False

        if old is not None:
            cpu_ary_exists = old.has_cpu_ary
            gpu_ary_exist = old.has_gpu_ary
            has_cpu_ary = has_cpu_ary or cpu_ary_exists or self.store_cpu is True
            has_gpu_ary = has_gpu_ary or gpu_ary_exists

        # Create a new record
        new = ArrayRecord(
            name=name,
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
        cpu_name = BaseSharedData.__cpu_name(name)
        gpu_name = BaseSharedData.__gpu_name(name)

        # Create descriptors on the class instance, even though members
        # may not necessarily be created on object instances. This is so
        # that if someone registers an array but doesn't ask for it to be
        # created, we have control over it, if at some later point they wish
        # to do a
        #
        # sd.blah_cpu = ...
        #
        setattr(BaseSharedData, cpu_name, CPUArrayDescriptor(record_key=name))
        setattr(BaseSharedData, gpu_name, GPUArrayDescriptor(record_key=name))

        # Create an empty cpu array if it doesn't exist
        # and set it on the object instance
        if cpu_ary_exists is not True and has_cpu_ary is True:
            cpu_ary = np.empty(shape=shape, dtype=dtype)
            setattr(self, cpu_name, cpu_ary)

        # Create an empty gpu array if it doesn't exist
        # and set it on the object instance
        # Also create a transfer method for tranferring data to the GPU
        if gpu_ary_exists is not True and has_gpu_ary is True:
            gpu_ary = gpuarray.empty(shape=shape, dtype=dtype)
            setattr(self, gpu_name, gpu_ary)

            # Create the transfer method
            def transfer(self, npary):
                self.check_array(name, npary)
                if self.store_cpu: setattr(self,cpu_name,npary)
                getattr(self,gpu_name).set(npary)

            # Create the method on ourself
            method_name = BaseSharedData.__transfer_method_name(name)
            method = types.MethodType(transfer,self)
            setattr(self,  method_name, method)
            # Create a docstring!
            getattr(method, '__func__').__doc__ = \
            """
            Transfers the npary numpy array to the %s gpuarray.
            npary and %s must be the same shape and type.
            """ % (gpu_name,gpu_name)

        # Set up a member describing the shape
        if kwargs.get('shape_member', False) is True:
            shape_name = BaseSharedData.__shape_name(name)
            setattr(self, shape_name, shape)

        # Set up a member describing the dtype
        if kwargs.get('dtype_member', False) is True:
            dtype_name = BaseSharedData.__dtype_name(name)
            setattr(self, dtype_name, dtype)

    def register_property(self, name, dtype, default, registrant, **kwargs):
        """
        Registers a property with this SharedData object

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

        self.properties[name] = pr = PropertyRecord(
            name, dtype, default, registrant)

        # Create the descriptor for this property on the class instance
        setattr(BaseSharedData, name, PropertyDescriptor(record_key=name, default=default))
        # Set the descriptor on this object instance
        setattr(self, name, default)

        # Should we create a setter for this property?
        setter = kwargs.get('setter', False)
        setter_name = BaseSharedData.__setter_name(name)

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

    def get_properties(self):
        """
        Returns a dictionary of properties related to this SharedData object.

        Used in templated GPU kernels.
        """
        sd = self
        
        D = {
            'na' : sd.na,
            'nbl' : sd.nbl,
            'nchan' : sd.nchan,
            'ntime' : sd.ntime,
            'npsrc' : sd.npsrc,
            'ngsrc' : sd.ngsrc,
            'nsrc'  : sd.nsrc,
            'nvis' : sd.nvis,
        }

        for p in self.properties.itervalues():
            D[p.name] = getattr(self,p.name)

        return D

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def get_default_ant_pairs(self):
        """
        Return an np.array(shape=(2, nbl, ntime), dtype=np.int32]) containing the
        default antenna pairs for each baseline at each timestep.
        """
        # Create the antenna pair mapping, from upper triangle indices
        # based on the number of antenna. 
        sd = self

        tmp = np.int32(np.triu_indices(sd.na,1))
        tmp = np.tile(tmp,sd.ntime).reshape(2,sd.ntime,sd.nbl)
        tmp = np.rollaxis(tmp, axis=2, start=1)
        return tmp.copy()

    def gen_array_descriptions(self):
        """ Generator generating strings describing each registered array """
        yield 'Registered Arrays'
        yield '-'*80
        yield BaseSharedData.fmt_array_line('Array Name','Size','Type','CPU','GPU','Shape')
        yield '-'*80

        for a in self.arrays.itervalues():
            yield BaseSharedData.fmt_array_line(a.name,
                BaseSharedData.fmt_bytes(BaseSharedData.array_bytes(a.shape, a.dtype)),
                np.dtype(a.dtype).name,
                'Y' if a.has_cpu_ary else 'N',
                'Y' if a.has_gpu_ary else 'N',
                a.shape)

    def gen_property_descriptions(self):
        """ Generator generating string describing each registered property """
        yield 'Registered Properties'
        yield '-'*80
        yield BaseSharedData.fmt_property_line('Property Name',
            'Type', 'Value', 'Default Value')
        yield '-'*80

        for p in self.properties.itervalues():
            yield BaseSharedData.fmt_property_line(
                p.name, np.dtype(p.dtype).name,
                getattr(self, p.name), p.default)

    def __str__(self):
        """ Outputs a string representation of this object """
        n_cpu_bytes = np.sum([BaseSharedData.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues() if a.has_cpu_ary is True])

        n_gpu_bytes = np.sum([BaseSharedData.array_bytes(a.shape,a.dtype)
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
            '%-*s: %s' % (w,'CPU Memory', BaseSharedData.fmt_bytes(n_cpu_bytes)),
            '%-*s: %s' % (w,'GPU Memory', BaseSharedData.fmt_bytes(n_gpu_bytes))]

        l.extend([''])
        l.extend([s for s in self.gen_array_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)