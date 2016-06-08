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

import numpy as np
import types

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

import montblanc
from hypercube import HyperCube
from montblanc.config import RimeSolverConfig as Options

import montblanc.util as mbu

class RIMESolver(HyperCube):
    def __init__(self, *args, **kwargs):
        """
        RIMESolver Constructor

        Keyword Arguments
        ---------
            slvr_cfg : dictionary
                Contains configuration options for this solver
        """

        super(RIMESolver, self).__init__(*args, **kwargs)
        # Store the solver configuration

        self._slvr_cfg = slvr_cfg = kwargs.get('slvr_cfg', None)

        if slvr_cfg is None:
            raise ValueError("Expected a 'slvr_cfg' keyword.")

        # Configure our floating point and complex types
        if slvr_cfg[Options.DTYPE] == Options.DTYPE_FLOAT:
            self.ft = np.float32
            self.ct = np.complex64
        elif slvr_cfg[Options.DTYPE] == Options.DTYPE_DOUBLE:
            self.ft = np.float64
            self.ct = np.complex128
        else:
            raise TypeError('Invalid dtype %s ' % slvr_cfg[Options.DTYPE])

        # Should we use the weight vector when computing the X2?
        self._use_weight_vector = slvr_cfg.get(Options.WEIGHT_VECTOR)

        # Is this solver handling auto-correlations
        self._is_auto_correlated = slvr_cfg.get(Options.AUTO_CORRELATIONS)

        # Is this solver outputting visibilities or residuals
        self._visibility_output = slvr_cfg.get(Options.VISIBILITY_OUTPUT)

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def use_weight_vector(self):
        return self._use_weight_vector

    def outputs_model_visibilities(self):
        return self._visibility_output == Options.VISIBILITY_OUTPUT_MODEL

    def outputs_residuals(self):
        return self._visibility_output == Options.VISIBILITY_OUTPUT_RESIDUALS

    def is_autocorrelated(self):
        """ Does this solver handle autocorrelations? """
        return self._is_auto_correlated == True

    def default_base_ant_pairs(self):
        """
        Return a list of two arrays containing the
        default antenna pairs for each baseline.
        """
        na = self.dim_local_size('na')
        k = 0 if self.is_autocorrelated() else 1
        return np.triu_indices(na, k)

    def register_default_dimensions(self):
        """ Register the default dimensions for a RIME solver """ 

        # Pull out the configuration options for the basics
        autocor = self._slvr_cfg.get(Options.AUTO_CORRELATIONS, False)
        ntime = self._slvr_cfg.get(Options.NTIME)
        na = self._slvr_cfg.get(Options.NA)
        nbands = self._slvr_cfg.get(Options.NBANDS)
        nchan = self._slvr_cfg.get(Options.NCHAN)
        npol = self._slvr_cfg.get(Options.NPOL)

        # Register these dimensions on this solver.
        self.register_dimension('ntime', ntime,
            description=Options.NTIME_DESCRIPTION)
        self.register_dimension('na', na,
            description=Options.NA_DESCRIPTION)
        self.register_dimension('nbands', nbands,
            description=Options.NBANDS_DESCRIPTION)
        self.register_dimension('nchan', nchan,
            description=Options.NCHAN_DESCRIPTION)
        self.register_dimension(Options.NPOL, npol,
            description=Options.NPOL_DESCRIPTION)

        # Now get the size of the registered dimensions
        ntime, na, nchan, npol = self.dim_local_size(
            'ntime', 'na', 'nchan', 'npol')
        
        nbl_expr = 'na*(na+1)//2' if autocor else 'na*(na-1)//2'
        nbl = self._slvr_cfg.get(Options.NBL, nbl_expr)

        self.register_dimension('nbl', nbl,
            description=Options.NBL_DESCRIPTION)

        self.register_dimension('npolchan', 'nchan*npol',
            description='Polarised channels')
        self.register_dimension('nvis', 'ntime*nbl*nchan',
            description='Visibilities')

        # Convert the source types, and their numbers
        # to their number variables and numbers
        # { 'point':10 } => { 'npsrc':10 }
        src_cfg = self._slvr_cfg[Options.SOURCES]
        src_nr_vars = mbu.sources_to_nr_vars(src_cfg)
        # Sum to get the total number of sources
        self.register_dimension('nsrc', sum(src_nr_vars.itervalues()),
            description=Options.NSRC_DESCRIPTION)

        # Register the individual source types
        for src_type, (nr_var, nr_of_src) in zip(
            src_cfg.iterkeys(), src_nr_vars.iteritems()):

            self.register_dimension(nr_var, nr_of_src, 
                description='{t} sources'.format(t=src_type),
                zero_valid=True)   

    def type_dict(self):
        """ Returns a dictionary mapping strings to concrete types """
        return {
            'ft' : self.ft,
            'ct' : self.ct,
            'int' : int,
        }

    def template_dict(self):
        """
        Returns a dictionary suitable for templating strings with
        properties and dimensions related to this Solver object.

        Used in templated GPU kernels.
        """
        slvr = self

        D = {
            # Constants
            'LIGHTSPEED': montblanc.constants.C,
        }

        # Map any types
        D.update(self.type_dict())

        # Update with dimensions
        D.update(self.dim_local_size_dict())

        # Add any registered properties to the dictionary
        for p in self._properties.itervalues():
            D[p.name] = getattr(self, p.name)

        return D

    def config(self):
        """ Returns the configuration dictionary for this solver """
        return self._slvr_cfg

    def register_array(self, name, shape, dtype, **kwargs):
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

        Returns
        -------
            A dictionary describing this array.
        """

        # Substitute any string types when calling the parent
        return super(RIMESolver, self).register_array(name, shape,
            mbu.dtype_from_str(dtype, self.type_dict()),
            **kwargs)

    def register_property(self, name, dtype, default, **kwargs):
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

        Returns
        -------
            A dictionary describing this property.

        """

        # Substitute any string types when calling the parent
        return super(RIMESolver, self).register_property(name,
            mbu.dtype_from_str(dtype, self.type_dict()),
            default, **kwargs)

    def create_arrays(self, ignore=None, supplied=None):
        """
        Create any necessary arrays on the solver. 

        Arguments
        ---------
            ignore : list
                List of array names to ignore.
            supplied : dictionary
                A dictionary of supplied arrays to create
                on the solver, keyed by name. Note that
                these arrays will not be initialised by
                montblanc, it is the responsibility of the
                user to initialise them.
        """
        raise NotImplementedError()

    @staticmethod
    def _arrays_to_create(reified_arrays, ignore, supplied):
        """
        Given reified_arrays, arrays to ignore and supplied arrays,
        work out which arrays must be created.
        """

        # Work out which arrays we shouldn't create
        dont_create = set(ignore)
        dont_create.update(supplied.iterkeys())

        return { n: a for n, a
            in reified_arrays.iteritems()
            if n not in dont_create }

    @staticmethod
    def _validate_supplied_arrays(reified_arrays, supplied):
        """
        Validate that the supplied arrays matched the shape
        and type of the reified arrays
        """

        for k, a in supplied.iteritems():
            expected_shape = reified_arrays[k].shape

            if a.shape != expected_shape:
                raise ValueError("Supplied array '{sn}'s' shape '{ss}' "
                    "does not match the expected shape of '{es}'".format(
                        sn=k, ss=a.shape, es=expected_shape))

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