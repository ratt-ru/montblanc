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
import pycuda.driver as cuda
import pycuda.gpuarray
import pycuda.tools
import types

import montblanc
import montblanc.util as mbu

from montblanc.BaseSolver import BaseSolver
from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

import montblanc.impl.biro.v4.BiroSolver as BSV4mod

from montblanc.impl.biro.v5.BiroSolver import BiroSolver

ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

ORDERING_CONSTRAINTS = { nr_var : 1 for nr_var in mbu.source_nr_vars() }
ORDERING_CONSTRAINTS.update({
    'nsrc' : 1,
    'ntime': 2,
    'nbl': 3,
    'na': 3,
    'nchan': 4
})

ORDERING_RANK = [' or '.join(['nsrc'] + mbu.source_nr_vars()),
    'ntime', ' or '.join(['nbl', 'na']), 'nchan']

def strip_and_create_virtual_source(ary_list, props):
    """
    This function strips out arrays associated with
    the various source types ('npsrc', 'ngsrc' etc.)
    and replaces them with a single array 'virtual_source'
    which is as large as the biggest of these sources. 
    """
    src_nr_vars = mbu.source_nr_vars()
    max_ary_bytes_per_src = 0
    remove_set = set()
    nsrc = props['nsrc']

    # Iterate over the array list, finding arrays specifically
    # related to the different source types. Find the most expensive
    # of these arrays, in terms of bytes per source
    for i, ary in enumerate(ary_list):
        # Figure out how many bytes this array uses in terms of
        # TOTAL sources 'nsrc'. This allows us to rank each
        # of the source arrays in terms of bytes per source.
        src_nrs = [nsrc for x in ary['shape'] if x in src_nr_vars]
        ary = ary.copy()
        ary['shape'] = tuple(['nsrc' if x in src_nr_vars else x for x in ary['shape']])

        if len(src_nrs) == 0:
            continue

        ary_bytes = mbu.dict_array_bytes(ary, props)
        total_src_nr = np.product(src_nrs)
        ary_bytes_per_src = ary_bytes / total_src_nr

        # Does this use the most bytes per source?
        if ary_bytes_per_src > max_ary_bytes_per_src:
            max_ary_bytes_per_src = ary_bytes_per_src

        # Mark this array for removal
        remove_set.add(i)

    # Create a new array list, removing source arrays
    # and creating a new virtual source array.
    new_ary_list = [v for i, v in enumerate(ary_list) if i not in remove_set]
    new_ary_list.append({
            'name': 'virtual_source',
            'dtype': np.int8,
            'shape' : (max_ary_bytes_per_src, 'nsrc'),
            'registrant' : 'BiroSolver',
        })

    return new_ary_list

class CompositeBiroSolver(BaseSolver):
    """
    Composite solver implementation for BIRO.

    Implements a solver composed of multiple BiroSolvers. The sub-solver
    memory transfers and pipelines are executed asynchronously.
    """
    def __init__(self, slvr_cfg):
        """
        BiroSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """

        # Set up a default pipeline if None is supplied

        super(CompositeBiroSolver, self).__init__(slvr_cfg)

        # Configure the dimensions of the beam cube
        self.beam_lw = self.slvr_cfg[Options.E_BEAM_WIDTH]
        self.beam_mh = self.slvr_cfg[Options.E_BEAM_HEIGHT]
        self.beam_nud = self.slvr_cfg[Options.E_BEAM_DEPTH]

        # Copy the v4 arrays and properties and
        # modify them for use on this
        # Composite Solver
        A_main, P_main = self.__twiddle_v4_arys_and_props(
            copy.deepcopy(BSV4mod.A), copy.deepcopy(BSV4mod.P))

        props = self.get_properties()
        # Strip out any arrays associated with the different
        # source types (point, Gaussian, sersic etc.) and
        # replace with a single virtual source array
        # which is big enough to hold any of them.
        A_sub = strip_and_create_virtual_source(A_main, props)
        P_sub = copy.deepcopy(BSV4mod.P)

        self.nsolvers = slvr_cfg.get('nsolvers', 4)
        self.dev_ctxs = slvr_cfg.get(Options.CONTEXT)
        self.solvers = []

        self.__validate_arrays(A_sub)

        if not isinstance(self.dev_ctxs, list):
            self.dev_ctxs = [self.dev_ctxs]

        for ctx in self.dev_ctxs:
            with mbu.ContextWrapper(ctx):
                # Query free memory on this context
                (free_mem,total_mem) = cuda.mem_get_info()

                # Work with a supplied memory budget, otherwise use
                # free memory less an amount equal to the upper size
                # of an NVIDIA context
                mem_budget = slvr_cfg.get('mem_budget', free_mem - 100*ONE_MB)

                # Figure out a viable dimension configuration
                # given the total problem size 
                viable, modded_dims = mbu.viable_dim_config(
                    mem_budget, A_sub, props,
                    ['nsrc=100', 'ntime', 'nbl=1&na=2','nchan=50%'],
                    self.nsolvers)                

                # Create property dictionary with update
                # dimensions.
                P = props.copy()
                P.update(modded_dims)

                if not viable:
                    dim_set_str = ', '.join(['%s=%s' % (k,v)
                        for k,v in modded_dims.iteritems()])

                    ary_list_str = '\n'.join(['%-*s %-*s %s' % (
                        15, a['name'],
                        10, mbu.fmt_bytes(mbu.dict_array_bytes(a, P)),
                        mbu.shape_from_str_tuple(a['shape'],P))
                        for a in sorted(A_sub,
                            reverse=True,
                            key=lambda a: mbu.dict_array_bytes(a, P))])

                    required_mem = mbu.dict_array_bytes_required(A_sub, P)

                    raise MemoryError("Tried reducing the problem size "
                        "by setting '%s' on all arrays, "
                        "but the resultant required memory of %s "
                        "for each of %d solvers is too big "
                        "to fit within the memory budget of %s. "
                        "List of biggests offenders:\n%s "
                        "\nSplitting the problem along the "
                        "channel dimension needs to be "
                        "implemented." %
                            (dim_set_str,
                            mbu.fmt_bytes(required_mem),
                            nsolvers,
                            mbu.fmt_bytes(mem_budget),
                            ary_list_str))

                # Create the configuration for the sub solver
                subslvr_cfg = BiroSolverConfiguration(**slvr_cfg)
                subslvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_DEFAULTS
                subslvr_cfg[Options.NA] = P[Options.NA]
                subslvr_cfg[Options.NTIME] = P[Options.NTIME]
                subslvr_cfg[Options.NCHAN] = P[Options.NCHAN]
                subslvr_cfg[Options.CONTEXT] = ctx

                # Extract the dimension differences
                self.src_diff = P['nsrc']
                self.time_diff = P[Options.NTIME]
                self.bl_diff = P['nbl']
                self.ant_diff = P[Options.NA]
                self.chan_diff = P[Options.NCHAN]

                # Create the sub-solvers for this context
                # and append
                for s in range(self.nsolvers):
                    subslvr = BiroSolver(subslvr_cfg)
                    # Our subsolvers don't think in terms
                    # of point, gaussian or other sources,
                    # but rather in terms of virtual sources
                    subslvr.twiddle_src_dims(P['nsrc'])
                    self.solvers.append(subslvr)

        assert len(self.solvers) == self.nsolvers*len(self.dev_ctxs)

        # Modify the array configuration for the sub-solvers
        # Don't create CPU arrays since we'll be copying them
        # from CPU arrays on the Composite Solver.
        # Do create GPU arrays, used for solving each sub-section
        # of the RIME.
        for ary in A_sub:
            # Add a transfer method
            ary['transfer_method'] = self.get_sub_transfer_method(ary['name'])
            ary['cpu'] = False
            ary['gpu'] = True

        # Create the arrays on the sub solvers
        for i, subslvr in enumerate(self.solvers):
            subslvr.register_properties(P_sub)
            subslvr.register_arrays(A_sub)

        self.use_weight_vector = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
        self.initialised=False

    def get_properties(self):
        # Obtain base solver property dictionary
        # and add the beam cube dimensions to it
        D = super(CompositeBiroSolver, self).get_properties()

        D.update({
            'beam_lw' : self.beam_lw,
            'beam_mh' : self.beam_mh,
            'beam_nud' : self.beam_nud
        })

        return D

    def __twiddle_v4_arys_and_props(self, arys, props):
        # Add a custom transfer method for transferring
        # arrays to the sub-solver. Also, in general,
        # only maintain CPU arrays on the main solver,
        # but not GPU arrays, which will exist on the sub-solvers
        for ary in arys:
            ary['transfer_method'] = self.get_transfer_method(ary['name'])
            ary['gpu'] = False
            ary['cpu'] = True

        # Add custom property setter method
        for prop in props:
            prop['setter_method'] = self.get_setter_method(ary['name'])

        # Do not create CPU versions of result arrays
        for ary in [a for a in arys if a['name'] in
                ['vis', 'B_sqrt', 'jones', 'chi_sqrd_result']]:
            ary['cpu'] = False

        return arys, props

    def transfer_arrays(self, sub_solver_idx, time_begin, time_end):
        """
        Transfer CPU arrays on the CompositeBiroSolver over to the
        BIRO sub-solvers asynchronously. A pinned memory pool is used
        to store the data temporarily.

        Arrays without a time dimension are transferred as a whole,
        otherwise the time section of the array appropriate to the
        sub-solver is transferred.
        """
        i = sub_solver_idx
        subslvr = self.solvers[i]
        stream = self.stream[i]

        for r in self.arrays.itervalues():
            # Is there anything to transfer for this array?
            if not r.has_cpu_ary:
                continue
            # Check for a time dimension. If the array has it,
            # we'll always be transferring a segment
            has_time = r.sshape.count('ntime') > 0
            # If this array was transferred and
            # has no time dimension, skip it
            if not has_time and subslvr.was_transferred[r.name]:
                continue

            cpu_name = mbu.cpu_name(r.name)
            gpu_name = mbu.gpu_name(r.name)

            # Get the CPU array on the composite solver
            # and the CPU array and the GPU array
            # on the sub-solver
            cpu_ary = getattr(self,cpu_name)
            gpu_ary = getattr(subslvr,gpu_name)

            # Set up the slicing of the main CPU array. It we're dealing with the
            # time dimension, slice the appropriate chunk, otherwise take
            # everything
            all_slice = slice(None,None,1)
            time_slice = slice(time_begin, time_end ,1)
            cpu_idx = tuple([time_slice if s == 'ntime' else all_slice
                for s in r.sshape])

            # Similarly, set up the slicing of the pinned CPU array
            time_diff = time_end - time_begin
            time_diff_slice = slice(None,time_diff,1)
            pin_idx = tuple([time_diff_slice if s == 'ntime' else all_slice
                for s in r.sshape])

            # Get a pinned array for asynchronous transfers
            cpu_shape_name = mbu.shape_name(r.name)
            cpu_dtype_name = mbu.dtype_name(r.name)

            pinned_ary = self.pinned_mem_pool.allocate(
                shape=getattr(subslvr,cpu_shape_name),
                dtype=getattr(subslvr,cpu_dtype_name))

            # Copy the numpy array into the pinned array
            # and perform the asynchronous transfer
            pinned_ary[pin_idx] = cpu_ary[cpu_idx]

            with self.context as ctx:
                gpu_ary.set_async(pinned_ary,stream=stream)

            # If we're not dealing with an array with a time dimension
            # then the transfer happens in one go. Otherwise, we're transferring
            # chunks of the the array and we're only done when
            # we transfer the last chunk
            if r.sshape.count('ntime') == 0 or time_end == self.ntime:
                subslvr.was_transferred[r.name] = True

    def __enter__(self):
        """
        When entering a run-time context related to this solver,
        initialise and return it.
        """
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        """
        When exiting a run-time context related to this solver,
        also perform exit for the sub-solvers.
        """
        self.shutdown()

    def initialise(self):
        """ Initialise the sub-solver """

        return

        if not self.initialised:
            with self.context as ctx:
                for i, slvr in enumerate(self.solvers):
                    slvr.initialise()

                # Get the reduction kernel
                # loaded in and hot
                pycuda.gpuarray.sum(
                    self.solvers[0].chi_sqrd_result_gpu,
                    stream=self.stream[0])

            self.initialised = True

    def __validate_arrays(self, arrays):
        """
        Check that the array dimension ordering is correct
        """
        for A in arrays:
            order = [ORDERING_CONSTRAINTS[var]
                for var in A['shape'] if var in ORDERING_CONSTRAINTS]

            if not all([b >= a for a, b in zip(order, order[1:])]):
                raise ValueError(('Array %s does not follow '
                    'ordering constraints. Shape is %s, but '
                    'this does breaks the expecting ordering of %s ') % (
                        A['name'], A['shape'],
                        ORDERING_RANK))

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        with self.context:
            pass

    def shutdown(self):
        """ Shutdown the solver """
        with self.context as ctx:
            for slvr in self.solvers:
                slvr.shutdown()

    def get_setter_method(self,name):
        """
        Setter method for CompositeBiroSolver properties. Sets the property
        on sub-solvers.
        """

        def setter(self, value):
            setattr(self,name,value)
            for slvr in self.solvers:
                setter_method_name = mbu.setter_name(name)
                setter_method = getattr(slvr,setter_method_name)
                setter_method(value)

        return types.MethodType(setter,self)

    def get_sub_transfer_method(self,name):
        def f(self, npary):
            raise Exception, 'Its illegal to call set methods on the sub-solvers'
        return types.MethodType(f,self)

    def get_transfer_method(self, name):
        """
        Transfer method for CompositeBiroSolver arrays. Sets the cpu array
        on the CompositeBiroSolver and indicates that it hasn't been transferred
        to the sub-solver
        """
        cpu_name = mbu.cpu_name(name)
        def transfer(self, npary):
            self.check_array(name, npary)
            setattr(self, cpu_name, npary)
            # Indicate that this data has not been transferred to
            # the sub-solvers
            for slvr in self.solvers: slvr.was_transferred[name] = False

        return types.MethodType(transfer,self)

    # Take these methods from the v2 BiroSolver
    get_default_base_ant_pairs = \
        BSV4mod.BiroSolver.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BSV4mod.BiroSolver.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BSV4mod.BiroSolver.__dict__['get_ap_idx']
