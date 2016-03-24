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

import numpy as np
import types

from weakref import WeakKeyDictionary
from attrdict import AttrDict
from collections import OrderedDict

import montblanc
import montblanc.util as mbu

from montblanc.enums import DIMDATA

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

class BaseSolver(object):
    """ Base class for solving the RIME. """

    def __init__(self):
        """
        BaseSolver Constructor
        """

        # Dictionaries to store records about our
        # dimensions, arrays and properties
        self._dims = OrderedDict()
        self._arrays = OrderedDict()
        self._properties = OrderedDict()

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([self.array_bytes(a) for a in self._arrays.itervalues()])

    def array_bytes(self, array):
        """ Estimates the memory of the supplied array in bytes """
        return np.product(array.shape)*np.dtype(array.dtype).itemsize

    def fmt_bytes(self, nbytes):
        """ Returns a human readable string, given the number of bytes """
        for x in ['B','KB','MB','GB', 'TB']:
            if nbytes < 1024.0:
                return "%3.1f%s" % (nbytes, x)
            nbytes /= 1024.0

        return "%.1f%s" % (nbytes, 'PB')

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return self.fmt_bytes(self.bytes_required())

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

        Returns
        -------
        A dictionary describing this dimension
        """
        from montblanc.enums import DIMDATA

        if name in self._dims:
            raise AttributeError((
                "Attempted to register dimension '{n}'' "
                "as an attribute of the solver, but "
                "it already exists. Please choose "
                "a different name!").format(n=name))

        # Create the dimension dictionary
        D = self._dims[name] = mbu.create_dim_data(name, dim_data, **kwargs)

        return D

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

    def dim_global_size_dict(self):
        """ Returns a mapping of dimension name to global size """
        return { d.name: d.global_size for d in self._dims.itervalues() }

    def dim_local_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_local_size('ntime, 'nbl', 'nchan')
        
        or

        ntime, nbl, nchan, nsrc = slvr.dim_local_size('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DIMDATA.LOCAL_SIZE, *args)

    def dim_local_size_dict(self):
        """ Returns a mapping of dimension name to local size """
        return { d.name: d.local_size for d in self._dims.itervalues() }

    def dim_extents(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_extents('ntime, 'nbl', 'nchan')
        
        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_extents('ntime,nbl:nchan nsrc')
        """

        return self.__dim_attribute(DIMDATA.EXTENTS, *args)

    def dim_extents_dict(self):
        """ Returns a mapping of dimension name to extents """
        return { d.name: d.extents for d in self.__dims.itervalues() }

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

        Returns
        -------
            A dictionary describing this array.
        """
        # Get a template dictionary to perform string replacements
        T = self.dim_local_size_dict()

        # Figure out the actual integer shape
        sshape = shape
        shape = mbu.shape_from_str_tuple(sshape, T)

        # Set up a member describing the shape
        if kwargs.get('shape_member', False) is True:
            shape_name = mbu.shape_name(name)
            setattr(self, shape_name, shape)

        # Set up a member describing the dtype
        if kwargs.get('dtype_member', False) is True:
            dtype_name = mbu.dtype_name(name)
            setattr(self, dtype_name, dtype)

        # Complain if array exists
        if name in self._arrays:
            raise ValueError(('Array %s is already registered '
                'on this solver object.') % name)

        # OK, create a record for this array
        A = self._arrays[name] = AttrDict(name=name,
            dtype=dtype, shape=shape, sshape=sshape,
            registrant=registrant, **kwargs)

        return A

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
        if name in self._properties:
            raise ValueError(('Property %s is already registered '
                'on this solver object.') % name)

        P = self._properties[name] = AttrDict(name=name, dtype=dtype,
            default=default, registrant=registrant)

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

        return P

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
        pass

    def initialise(self):
        """ Initialise the RIME solver """
        pass

    def shutdown(self):
        """ Stop the RIME solver """
        pass

    def __enter__(self):
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __str__(self):
        """ Outputs a string representation of this object """

        w = 20

        l = ['',
            '%-*s: %s' % (w,'Memory Usage', self.mem_required()),
            '-'*80]

        l.extend([''])
        l.extend([s for s in self.gen_dimension_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_array_descriptions()])
        l.extend([''])
        l.extend([s for s in self.gen_property_descriptions()])

        return '\n'.join(l)
