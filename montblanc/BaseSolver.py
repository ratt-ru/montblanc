#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import collections
import copy

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

import numpy as np
import types

from weakref import WeakKeyDictionary
from attrdict import AttrDict
from collections import OrderedDict

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc
import montblanc.factory
import montblanc.util as mbu

from montblanc.enums import (DIMDATA)

from montblanc.config import (BiroSolverConfigurationOptions as Options)

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
        dtype = instance._properties[self.record_key].dtype
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

class Solver(object):
    """ Base class for solving the RIME. """
    pass

class BaseSolver(Solver):
    """ Class that holds the elements for solving the RIME """

    pipeline = PipelineDescriptor()

    def __init__(self, slvr_cfg):
        """
        BaseSolver Constructor
        Arguments:
            slvr_cfg : SolverConfiguration
                A solver configuration object.
        """

        # Dictionaries to store records about our
        # dimensions, arrays and properties
        self._dims = OrderedDict()
        self._arrays = OrderedDict()
        self._properties = OrderedDict()

        # Store the solver configuration
        self._slvr_cfg = slvr_cfg

        # Configure our floating point and complex types
        if slvr_cfg[Options.DTYPE] == Options.DTYPE_FLOAT:
            self.ft = np.float32
            self.ct = np.complex64
        elif slvr_cfg[Options.DTYPE] == Options.DTYPE_DOUBLE:
            self.ft = np.float64
            self.ct = np.complex128
        else:
            raise TypeError('Invalid dtype %s ' % slvr_cfg[Options.DTYPE])

        # Store the context, choosing the default if not specified
        ctx = slvr_cfg.get(Options.CONTEXT, None)

        if ctx is None:
            raise Exception('No CUDA context was supplied to the BaseSolver')

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

        # Should we store CPU versions of the GPU arrays
        self.store_cpu = slvr_cfg.get(Options.STORE_CPU, False)

        # Is this a master solver or not
        self._is_master = slvr_cfg.get(Options.SOLVER_TYPE)

        # Configure our solver pipeline
        pipeline = slvr_cfg.get('pipeline', None)

        if pipeline is None:
            pipeline = montblanc.factory.get_empty_pipeline(slvr_cfg)
        self.pipeline = pipeline

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self._arrays.itervalues()])

    def cpu_bytes_required(self):
        """ returns the memory required by all CPU arrays in bytes. """
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self._arrays.itervalues() if a.cpu])

    def gpu_bytes_required(self):
        """ returns the memory required by all GPU arrays in bytes. """
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self._arrays.itervalues() if a.gpu])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return mbu.fmt_bytes(self.bytes_required())

    def register_dimension(self, name, dim_data, **kwargs):
        """
        Registers a dimension with this Solver object
        Arguments
        ---------
            dim_data : integer or dict
        Keyword Arguments
        -----------------
            description : string
                The description for this dimension.
                e.g. 'Number of timesteps'.
            global_size : integer
                The global size of this dimension across
                all solvers.
            local_size : integer or None
                The local size of this dimension
                on this solver. If None, set to
                the global_size.
            extents : list or tuple of length 2
                The extent of the dimension on the solver.
                E[0] < E[1] <= local_size must hold.
            zero_valid : boolean
                If True, this dimension may be zero-sized.
        """
        from montblanc.enums import DIMDATA

        if name in self._dims:
            raise AttributeError((
                "Attempted to register dimension '{n}'' "
                "as an attribute of the solver, but "
                "it already exists. Please choose "
                "a different name!").format(n=name))

        # Create the dimension dictionary
        self._dims[name] = mbu.create_dim_data(name, dim_data, **kwargs)

    def register_default_dimensions(self):
        """ Register the default dimensions for a RIME solver """ 

        # Pull out the configuration options for the basics
        autocor = self._slvr_cfg.get(Options.AUTO_CORRELATIONS, False)
        ntime = self._slvr_cfg.get(Options.NTIME)
        na = self._slvr_cfg.get(Options.NA)
        nchan = self._slvr_cfg.get(Options.NCHAN)
        npol = self._slvr_cfg.get(Options.NPOL)

        # Register these dimensions on this solver.
        self.register_dimension('ntime', ntime,
            description=Options.NTIME_DESCRIPTION)
        self.register_dimension('na', na,
            description=Options.NA_DESCRIPTION)
        self.register_dimension('nchan', nchan,
            description=Options.NCHAN_DESCRIPTION)
        self.register_dimension(Options.NPOL, npol,
            description=Options.NPOL_DESCRIPTION)

        size_func = (self.dim_global_size if self.is_master()
            else self.dim_local_size)

        # Now get the size of the registered dimensions
        ntime, na, nchan, npol = size_func('ntime', 'na', 'nchan', 'npol')
        
        nbl = self._slvr_cfg.get(Options.NBL,
            mbu.nr_of_baselines(na, autocor))

        self.register_dimension('nbl', nbl,
            description=Options.NBL_DESCRIPTION)

        nbl = size_func('nbl')

        self.register_dimension('npolchan', nchan*npol,
            description='Number of channels x polarisations')
        self.register_dimension('nvis', ntime*nbl*nchan,
            description='Number of visibilities')

        # Convert the source types, and their numbers
        # to their number variables and numbers
        # { 'point':10 } => { 'npsrc':10 }
        src_cfg = self._slvr_cfg[Options.SOURCES]
        src_nr_vars = mbu.sources_to_nr_vars(src_cfg)
        # Sum to get the total number of sources
        self.register_dimension('nsrc', sum(src_nr_vars.itervalues()),
            description='Number of sources (total)')

        # Register the individual source types
        for src_type, (nr_var, nr_of_src) in zip(
            src_cfg.iterkeys(), src_nr_vars.iteritems()):

            self.register_dimension(nr_var, nr_of_src, 
                description='Number of {t} sources'.format(t=src_type),
                zero_valid=True)        

    def register_dimensions(self, dim_list, defaults=True):
        for dim in dim_list:
            self.register_dimension(dim)

    def update_dimensions(self, dim_list):
        """
        >>> slvr.update_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """
        for dim_data in dim_list:
            self.update_dimension(dim_data)


    def update_dimension(self, update_dict):
        """
        Update the dimension size and extents.
        Arguments
        ---------
            update_dict : dict
        """
        name = update_dict.get(DIMDATA.NAME, None)

        if not name:
            raise AttributeError("A dimension name is required to update "
                "a dimension. Update dictionary {u}."
                    .format(u=update_dict))

        dim = self._dims.get(name, None)

        # Sanity check dimension existence
        if not dim:
            montblanc.log.warn("'Dimension {n}' cannot be updated as it "
                "is not registered in the dimension dictionary."
                    .format(n=name))

            return

        dim.update(update_dict)

    def __dim_attribute(self, attr, *args):
        """
        Returns a list of dimension attribute attr, for the
        dimensions specified as strings in args.
        ntime, nbl, nchan = slvr.__dim_attribute('global_size', ntime, 'nbl', 'nchan')
        
        or
        ntime, nbl, nchan, nsrc = slvr.__dim_attribute('global_size', 'ntime,nbl:nchan nsrc')
        """

        import re

        # If we got a single string argument
        if len(args) == 1 and type(args[0]) is str:

            result = [self._dims[name][attr] for name in
                [s.strip() for s in re.split(',|:|;| ', args[0])]]
        else:
            result = [self._dims[name][attr] for name in args]

        # Return single element if length one else entire list
        return result[0] if len(result) == 1 else result

    def dim_global_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_global_size('ntime, 'nbl', 'nchan')
        
        or
        ntime, nbl, nchan, nsrc = slvr.dim_global_size('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DIMDATA.GLOBAL_SIZE, *args)

    def dim_local_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_local_size('ntime, 'nbl', 'nchan')
        
        or
        ntime, nbl, nchan, nsrc = slvr.dim_local_size('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DIMDATA.LOCAL_SIZE, *args)

    def dim_extents(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_extents('ntime, 'nbl', 'nchan')
        
        or
        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_extents('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DIMDATA.EXTENTS, *args)

    def check_array(self, record_key, ary):
        """
        Check that the shape and type of the supplied array matches
        our supplied record
        """
        record = self._arrays[record_key]

        if record.shape != ary.shape:
            raise ValueError(('%s\'s shape %s is different '
                'from the shape %s of the supplied argument.') %
                    (record.name, record.shape, ary.shape))

        if record.dtype != ary.dtype:
            raise TypeError(('%s\'s type \'%s\' is different '
                'from the type \'%s\' of the supplied argument.') % 
                    (record.name,
                    np.dtype(record.dtype).name,
                    np.dtype(ary.dtype).name))

    def init_array(self, name, ary, value):
        # No defaults are supplied
        if value is None:
            ary.fill(0)
        # The array is defaulted with some function
        elif isinstance(value, types.MethodType):
            try:
                signature(value).bind(self, ary)
            except TypeError:
                raise TypeError(('The signature of the function supplied '
                    'for setting the value on array %s is incorrect. '
                    'The function signature has the form f(slvr, ary), '
                    'where f is some function that will set values '
                    'on the array, slvr is a Solver object which provides ' 
                    'useful information to the function, '
                    'and ary is the NumPy array which must be '
                    'initialised with values.') % (name))

            returned_ary = value(self, ary)

            if returned_ary is not None:
                ary[:] = returned_ary
        elif isinstance(value, types.LambdaType):
            try:
                signature(value).bind(self, ary)
            except TypeError:
                raise TypeError(('The signature of the lambda supplied '
                    'for setting the value on array %s is incorrect. '
                    'The function signature has the form lambda slvr, ary:, '
                    'where lambda provides functionality for setting values '
                    'on the array, slvr is a Solver object which provides ' 
                    'useful information to the function, '
                    'and ary is the NumPy array which must be '
                    'initialised with values.') % (name))

            returned_ary = value(self, ary)

            if returned_ary is not None:
                ary[:] = returned_ary
        # Got an ndarray, try set it equal
        elif isinstance(value, np.ndarray):
            try:
                ary[:] = value
            except BaseException as e:
                raise ValueError(('Tried to assign array %s with '
                    'value NumPy array, but this failed '
                    'with %s') % (name, repr(e)))
        # Assume some sort of value has been supplied
        # Give it to NumPy
        else:
            try:
                ary.fill(value)
            except BaseException as e:
                raise ValueError(('Tried to fill array %s with '
                    'value value %s, but NumPy\'s fill function '
                    'failed with %s') % (name, value, repr(e)))


    def register_array(self, name, shape, dtype, registrant, **kwargs):
        """
        Register an array with this Solver object.
        Arguments
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
            transfer_method : boolean or function
                True by default. If True, a default 'transfer_name' member is
                created, otherwise the supplied function is used instead
            page_locked : boolean
                True if the 'name_cpu' ndarray should be allocated as
                a page-locked array.
            aligned : boolean
                True if the 'name_cpu' ndarray should be allocated as
                an page-aligned array.
            replace : boolean
                True if existing arrays should be replaced.
        """
        # Should we create arrays? By default we don't create CPU arrays
        # but we do create GPU arrays by default.
        create_cpu_ary = kwargs.get('cpu', False) or self.store_cpu
        create_gpu_ary = kwargs.get('gpu', True)

        # Get a template dictionary to perform string replacements
        T = self.template_dict()

        # Figure out the actual integer shape
        sshape = shape
        shape = mbu.shape_from_str_tuple(sshape, T)

        # Replace any string representations with the
        # appropriate data type
        dtype = mbu.dtype_from_str(dtype, T)

        # OK, create a record for this array
        if name not in self._arrays:
            self._arrays[name] = AttrDict(name=name, dtype=dtype,
                shape=shape, sshape=sshape,
                registrant=registrant, **kwargs)
        else:
            raise ValueError(('Array %s is already registered '
                'on this solver object.') % name)

        # Attribute names
        cpu_name = mbu.cpu_name(name)
        gpu_name = mbu.gpu_name(name)

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

        # If we're creating arrays, then we'll want to initialise
        # them with default values
        default_ary = None

        # If we're creating test data, initialise the array with
        # data from the test key, otherwise take data from the default key
        if self._slvr_cfg[Options.DATA_SOURCE] == Options.DATA_SOURCE_TEST:
            source_key = 'test'
        else:
            source_key = 'default'

        if create_cpu_ary or create_gpu_ary:
            page_locked = kwargs.get('page_locked', False)
            aligned = kwargs.get('aligned', False)

            with self.context:
                # Page locked implies aligned
                if page_locked:
                    default_ary = cuda.pagelocked_empty(shape=shape, dtype=dtype)
                elif aligned:
                    default_ary = cuda.aligned_empty(shape=shape, dtype=dtype)
                else:
                    default_ary = np.empty(shape=shape, dtype=dtype)
                    
                self.init_array(name, default_ary, kwargs.get(source_key, None))

        # Create an empty cpu array if it doesn't exist
        # and set it on the object instance
        if create_cpu_ary:
            setattr(self, cpu_name, default_ary)

        # Create an empty gpu array if it doesn't exist
        # and set it on the object instance
        # Also create a transfer method for tranferring data to the GPU
        if create_gpu_ary:
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
                        s=mbu.fmt_bytes(mbu.array_bytes(shape, dtype)),
                        f=mbu.fmt_bytes(free_mem),
                        t=mbu.fmt_bytes(total_mem)))

                gpu_ary = gpuarray.empty(shape=shape, dtype=dtype)

                # If the array length is non-zero initialise it
                if np.product(shape) > 0:
                    # If available, use CPU defaults
                    # to initialise the array
                    if create_cpu_ary:
                        gpu_ary.set(default_ary)
                    # Otherwise just zero it 
                    else:
                        gpu_ary.fill(0)
                
                setattr(self, gpu_name, gpu_ary)

        # Should we create a setter for this property?
        transfer_method = kwargs.get('transfer_method', True)

        # OK, we got a boolean for the kwarg, create a default transfer method
        if isinstance(transfer_method, types.BooleanType) and transfer_method is True:
            # Create the transfer method
            def transfer(self, npary):
                self.check_array(name, npary)
                if create_cpu_ary:
                    setattr(self,cpu_name,npary)
                if create_gpu_ary:
                    with self.context:
                        getattr(self,gpu_name).set(npary)

            transfer_method = types.MethodType(transfer,self)
        # Otherwise, we can just use the supplied kwarg
        elif isinstance(transfer_method, types.MethodType):
            pass
        else:
            raise TypeError(('transfer_method keyword argument set '
                'to an invalid type %s') % (type(transfer_method)))

        # Name the transfer method
        transfer_method_name = mbu.transfer_method_name(name)
        setattr(self,  transfer_method_name, transfer_method)
        # Create a docstring!
        getattr(transfer_method, '__func__').__doc__ = \
        """
        Transfers the npary numpy array to the %s gpuarray.
        npary and %s must be the same shape and type.
        """ % (gpu_name,gpu_name)


        # Set up a member describing the shape
        if kwargs.get('shape_member', False) is True:
            shape_name = mbu.shape_name(name)
            setattr(self, shape_name, shape)

        # Set up a member describing the dtype
        if kwargs.get('dtype_member', False) is True:
            dtype_name = mbu.dtype_name(name)
            setattr(self, dtype_name, dtype)

    def register_arrays(self, array_list):
        """
        Register arrays using a list of dictionaries defining the arrays.
        The list should itself contain dictionaries. i.e.
        >>> D = [
            'uvw' : { 'name':'uvw', 'shape':(3,'ntime','nbl'),'dtype':np.float32 },
            'lm' : { 'name':'lm', 'shape':(2,'nsrc'),'dtype':np.float32 }
        ]
        """
        for ary in array_list:
            self.register_array(**ary)

    def register_property(self, name, dtype, default, registrant, **kwargs):
        """
        Registers a property with this Solver object
        Arguments
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

        T = self.template_dict()

        # Replace any string representations with the
        # appropriate data type
        dtype = mbu.dtype_from_str(dtype, T)

        if name not in self._properties:
            self._properties[name] = AttrDict(name=name, dtype=dtype,
                default=default, registrant=registrant)
        else:
            raise ValueError(('Property %s is already registered '
                'on this solver object.') % name)

        #if not hasattr(BaseSolver, name):
        if not BaseSolver.__dict__.has_key(name):
                # Create the descriptor for this property on the class instance
            setattr(BaseSolver, name, PropertyDescriptor(record_key=name, default=default))

        # Set the descriptor on this object instance
        setattr(self, name, default)

        # Should we create a setter for this property?
        setter = kwargs.get('setter_method', True)
        setter_name = mbu.setter_name(name)

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
            setattr(self, setter_name, setter)
        else:
            raise TypeError, ('setter keyword argument set',
                ' to an invalid type %s' % (type(setter)))

    def register_properties(self, property_list):
        """
        Register properties using a list defining the properties.
        The dictionary should itself contain dictionaries. i.e.
        >>> D = [
            { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'solver' },
        ]
        """
        for prop in property_list:
            self.register_property(**prop)

    def template_dict(self):
        """
        Returns a dictionary suitable for templating strings with
        properties and dimensions related to this Solver object.
        Used in templated GPU kernels.
        """
        slvr = self

        D = {
            # Types
            'ft' : slvr.ft,
            'ct' : slvr.ct,
            'int' : int,
            # Constants
            'LIGHTSPEED': montblanc.constants.C,
        }

        # Update with dimensions
        D.update({d.name: d.local_size for d in self._dims.itervalues()})

        # Add any registered properties to the dictionary
        for p in self._properties.itervalues():
            D[p.name] = getattr(self, p.name)

        return D

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def is_master(self):
        """ Is this a master solver """
        return self._is_master == Options.SOLVER_TYPE_MASTER

    def properties(self):
        """ Returns a dictionary of properties """
        return self._properties

    def property(self, name):
        """ Returns a property """
        try:
            return self._properties[name]
        except KeyError:
            raise KeyError("Property '{n}' is not registered "
                "on this solver".format(n=name))

    def arrays(self):
        """ Returns a dictionary of arrays """
        return self._arrays

    def array(self, name):
        """ Returns an array """
        try:
            return self._arrays[name]
        except KeyError:
            raise KeyError("Array '{n}' is not registered "
                "on this solver".format(n=name))

    def dimensions(self):
        """ Return a dictionary of dimensions """
        return self._dims

    def dimension(self, name):
        """ Returns a dimension """
        try:
            return self._dims[name]
        except KeyError:
            raise KeyError("Array '{n}' is not registered "
                "on this solver".format(n=name))

    def gen_dimension_descriptions(self):
        """ Generator generating string describing each registered dimension """
        yield 'Registered Dimensions'
        yield '-'*80
        yield mbu.fmt_dimension_line('Dimension Name', 'Description', 'Size')
        yield '-'*80

        for d in sorted(self._dims.itervalues(), key=lambda x: x.name.upper()):
            yield mbu.fmt_dimension_line(
                d.name, d.description, d.local_size)

    def gen_array_descriptions(self):
        """ Generator generating strings describing each registered array """
        yield 'Registered Arrays'
        yield '-'*80
        yield mbu.fmt_array_line('Array Name','Size','Type','CPU','GPU','Shape')
        yield '-'*80

        for a in sorted(self._arrays.itervalues(), key=lambda x: x.name.upper()):
            yield mbu.fmt_array_line(a.name,
                mbu.fmt_bytes(mbu.array_bytes(a.shape, a.dtype)),
                np.dtype(a.dtype).name,
                'Y' if a.cpu else 'N',
                'Y' if a.gpu else 'N',
                a.sshape)

    def gen_property_descriptions(self):
        """ Generator generating string describing each registered property """
        yield 'Registered Properties'
        yield '-'*80
        yield mbu.fmt_property_line('Property Name',
            'Type', 'Value', 'Default Value')
        yield '-'*80

        for p in sorted(self._properties.itervalues(), key=lambda x: x.name.upper()):
            yield mbu.fmt_property_line(
                p.name, np.dtype(p.dtype).name,
                getattr(self, p.name), p.default)

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

    def __enter__(self):
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __str__(self):
        """ Outputs a string representation of this object """
        n_cpu_bytes = self.cpu_bytes_required()
        n_gpu_bytes = self.gpu_bytes_required()

        w = 20

        l = ['',
            'Memory Usage',
            '-'*80,
            '%-*s: %s' % (w,'CPU Memory', mbu.fmt_bytes(n_cpu_bytes)),
            '%-*s: %s' % (w,'GPU Memory', mbu.fmt_bytes(n_gpu_bytes))]

        l.extend([''])
        l.extend([s for s in self.gen_dimension_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_array_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)
