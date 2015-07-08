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

import copy
import numpy as np
import types

from weakref import WeakKeyDictionary

import pycuda
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

import montblanc
import montblanc.factory
import montblanc.util as mbu

from montblanc.config import (BiroSolverConfigurationOptions as Options)

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
    """ Base class for solving the RIME. """
    pass

DEFAULT_NA=3
DEFAULT_NBL=mbu.nr_of_baselines(DEFAULT_NA)
DEFAULT_NCHAN=4
DEFAULT_NTIME=10
DEFAULT_NVIS=DEFAULT_NBL*DEFAULT_NCHAN*DEFAULT_NTIME
DEFAULT_NSRC=1

class BaseSolver(Solver):
    """ Class that holds the elements for solving the RIME """
    na = Parameter(DEFAULT_NA)
    nbl = Parameter(DEFAULT_NBL)
    nchan = Parameter(DEFAULT_NCHAN)
    ntime = Parameter(DEFAULT_NTIME)
    nsrc = Parameter(DEFAULT_NSRC)
    nvis = Parameter(DEFAULT_NVIS)

    pipeline = PipelineDescriptor()

    def __init__(self, slvr_cfg):
        """
        BaseSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
        """

        super(BaseSolver, self).__init__()

        autocor = slvr_cfg.get(Options.AUTO_CORRELATIONS, False)

        # Configure our problem dimensions. Number of
        # - antenna
        # - baselines
        # - channels
        # - timesteps
        # - point sources
        # - gaussian sources
        # - sersic sources
        self.na = slvr_cfg[Options.NA]
        self.nbl = nbl = mbu.nr_of_baselines(self.na,autocor)
        self.nchan = slvr_cfg[Options.NCHAN]
        self.ntime = slvr_cfg[Options.NTIME]
        self.nvis = self.nbl*self.nchan*self.ntime

        src_nr_vars = mbu.sources_to_nr_vars(slvr_cfg[Options.SOURCES])
        self.nsrc = sum(src_nr_vars.itervalues())

        for nr_var, nr_of_src in src_nr_vars.iteritems():
            setattr(self, nr_var, nr_of_src)

        if self.nsrc == 0:
            raise ValueError('The number of sources, or, ',
                            'the sum of %s, '
                            'must be greater than zero') % \
                            (src_nr_vars)

        # Configure our floating point and complex types
        if slvr_cfg[Options.DTYPE] == Options.DTYPE_FLOAT:
            self.ft = np.float32
            self.ct = np.complex64
        elif slvr_cfg[Options.DTYPE] == Options.DTYPE_DOUBLE:
            self.ft = np.float64
            self.ct = np.complex128
        else:
            raise TypeError, ('Invalid dtype %s ' % slvr_cfg[Options.DTYPE])

        # Store the context, choosing the default if not specified
        ctx = slvr_cfg.get('context', None)

        if ctx is None:
            raise Exception, 'No context was supplied to the BaseSolver'

        # Create a context wrapper
        self.context = mbu.ContextWrapper(ctx)

        # Figure out the integer compute cability of the device
        # associated with the context
        with self.context as ctx:
            cc_tuple = cuda.Context.get_device().compute_capability()
        # np.dot((3,5), (100,10)) = 3*100 + 5*10 = 350 for Kepler
        self.cc = np.int32(np.dot(cc_tuple, (100,10)))

        # Dictionaries to store records about our arrays and properties
        self.arrays = {}
        self.properties = {}

        # Should we store CPU versions of the GPU arrays
        self.store_cpu = slvr_cfg.get('store_cpu', False)

        # Configure our solver pipeline
        pipeline = slvr_cfg.get('pipeline', None)

        if pipeline is None:
            pipeline = montblanc.factory.get_empty_pipeline()
        self.pipeline = pipeline

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues()])

    def cpu_bytes_required(self):
        """ returns the memory required by all CPU arrays in bytes. """
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues() if a.has_cpu_ary])

    def gpu_bytes_required(self):
        """ returns the memory required by all GPU arrays in bytes. """
        return np.sum([mbu.array_bytes(a.shape,a.dtype)
            for a in self.arrays.itervalues() if a.has_gpu_ary])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return mbu.fmt_bytes(self.bytes_required())

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
            transfer_method : boolean or function
                True by default. If True, a default 'transfer_name' member is
                created, otherwise the supplied function is used instead
            page_locked : boolean
                True if the 'name_cpu' ndarray should be allocated as
                a page-locked array
            replace : boolean
                True if existing arrays should be replaced.
        """
        # Try and find an existing version of this array
        old = self.arrays.get(name, None)

        # Should we create arrays? By default we don't create CPU arrays
        # but we do create GPU arrays by default.
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

        # Get a property dictionary to perform string replacements
        P = self.get_properties()

        # Figure out the actual integer shape
        sshape = shape
        shape = mbu.shape_from_str_tuple(sshape, P)

        # Replace any string representations with the
        # appropriate data type
        dtype = mbu.dtype_from_str(dtype, P)

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

        page_locked = kwargs.get('page_locked', False)

        # Create an empty cpu array if it doesn't exist
        # and set it on the object instance
        if create_cpu_ary:
            if not page_locked:
                cpu_ary = np.zeros(shape=shape, dtype=dtype)
            else:
                with self.context as ctx:
                    cpu_ary = pycuda.driver.pagelocked_zeros(
                        shape=shape, dtype=dtype)

            setattr(self, cpu_name, cpu_ary)

        # Create an empty gpu array if it doesn't exist
        # and set it on the object instance
        # Also create a transfer method for tranferring data to the GPU
        if create_gpu_ary:
            # We don't use gpuarray.zeros, since it fails for
            # a zero-length array. This is kind of bad since
            # the gpuarray returned by gpuarray.empty() doesn't
            # have GPU memory allocated to it.
            with self.context as ctx:
                gpu_ary = gpuarray.empty(shape=shape, dtype=dtype)

                # Zero the array, if it has non-zero length
                if np.product(shape) > 0: gpu_ary.fill(dtype(0))
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
            raise TypeError, ('transfer_method keyword argument set',
                ' to an invalid type %s' % (type(transfer_method)))

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

        P = self.get_properties()

        # Replace any string representations with the
        # appropriate data type
        dtype = mbu.dtype_from_str(dtype, P)

        self.properties[name]  = PropertyRecord(
            name, dtype, default, registrant)

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

    def get_array_record(self, name):
        return self.arrays[name]

    def get_properties(self):
        """
        Returns a dictionary of properties related to this Solver object.

        Used in templated GPU kernels.
        """
        slvr = self

        D = {
            # Dimensions
            'na' : slvr.na,
            'nbl' : slvr.nbl,
            'nchan' : slvr.nchan,
            'ntime' : slvr.ntime,
            'npsrc' : slvr.npsrc,
            'ngsrc' : slvr.ngsrc,
            'nssrc' : slvr.nssrc,
            'nsrc'  : slvr.nsrc,
            'nvis' : slvr.nvis,
            # Types
            'ft' : slvr.ft,
            'ct' : slvr.ct,
            'int' : int,
            # Constants
            'LIGHTSPEED': montblanc.constants.C,
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
        yield mbu.fmt_array_line('Array Name','Size','Type','CPU','GPU','Shape')
        yield '-'*80

        for a in sorted(self.arrays.itervalues(), key=lambda x: x.name.upper()):
            yield mbu.fmt_array_line(a.name,
                mbu.fmt_bytes(mbu.array_bytes(a.shape, a.dtype)),
                np.dtype(a.dtype).name,
                'Y' if a.has_cpu_ary else 'N',
                'Y' if a.has_gpu_ary else 'N',
                a.sshape)

    def gen_property_descriptions(self):
        """ Generator generating string describing each registered property """
        yield 'Registered Properties'
        yield '-'*80
        yield mbu.fmt_property_line('Property Name',
            'Type', 'Value', 'Default Value')
        yield '-'*80

        for p in sorted(self.properties.itervalues(), key=lambda x: x.name.upper()):
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
            'RIME Dimensions',
            '-'*80,
            '%-*s: %s' % (w,'Antenna', self.na),
            '%-*s: %s' % (w,'Baselines', self.nbl),
            '%-*s: %s' % (w,'Channels', self.nchan),
            '%-*s: %s' % (w,'Timesteps', self.ntime),
            '%-*s: %s' % (w,'Point Sources', self.npsrc),
            '%-*s: %s' % (w,'Gaussian Sources', self.ngsrc),
            '%-*s: %s' % (w,'Sersic Sources', self.nssrc),
            '%-*s: %s' % (w,'CPU Memory', mbu.fmt_bytes(n_cpu_bytes)),
            '%-*s: %s' % (w,'GPU Memory', mbu.fmt_bytes(n_gpu_bytes))]

        l.extend([''])
        l.extend([s for s in self.gen_array_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)
