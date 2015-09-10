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
ORDERING_CONSTRAINTS.update({ 'nsrc' : 1,
    'ntime': 2, 'nbl': 3, 'na': 3, 'nchan': 4 })

ORDERING_RANK = [' or '.join(['nsrc'] + mbu.source_nr_vars()),
    'ntime', ' or '.join(['nbl', 'na']), 'nchan']

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

        self.register_properties(P_main)
        self.register_arrays(A_main)

        props = self.get_properties()
        A_sub = copy.deepcopy(BSV4mod.A)
        P_sub = copy.deepcopy(BSV4mod.P)

        self.nsolvers = slvr_cfg.get('nsolvers', 4)
        self.dev_ctxs = slvr_cfg.get(Options.CONTEXT)
        self.solvers = []
        self.stream = []

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
                    # Configure the total number of sources
                    # handled by each sub-solver
                    subslvr.cfg_total_src_dims(P['nsrc'])
                    self.solvers.append(subslvr)
                    self.stream.append(cuda.Stream())

        assert len(self.solvers) == self.nsolvers*len(self.dev_ctxs)

        A_sub, P_sub = self.__twiddle_v4_subarys_and_props(A_sub, P_sub)

        # Create the arrays on the sub solvers
        for i, subslvr in enumerate(self.solvers):
            subslvr.register_properties(P_sub)
            subslvr.register_arrays(A_sub)

        self.use_weight_vector = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
        self.initialised=False

    def __gen_rime_slices(self):
        nr_vars = ['ntime', 'nbl', 'na', 'na1', 'nchan', 'nsrc']
        src_nr_var_counts = mbu.sources_to_nr_vars(
            self.slvr_cfg[Options.SOURCES])
        src_nr_vars = mbu.source_nr_vars()
        nr_vars.extend(src_nr_vars)

        # Create the slice dictionaries, which we use to index
        # dimensions of the CPU and GPU array.
        # gpu_slice arrays generally start at 0 and stop
        # at the associated associated cpu_slice array length
        cpu_slice = {v: slice(None, None, 1) for v in nr_vars}
        gpu_slice = {v: slice(None, None, 1) for v in nr_vars}
        # gpu_count is explicitly set to this length
        gpu_count = {v: 0 for v in nr_vars if v != 'na1'}

        # Set up time slicing
        for t in xrange(0, self.ntime, self.time_diff):
            t_end = min(t + self.time_diff, self.ntime)
            t_diff = t_end - t
            cpu_slice['ntime'] = slice(t,  t_end, 1)
            gpu_slice['ntime'] = slice(0, t_diff, 1)
            gpu_count['ntime'] = t_diff

            # Set up baseline and antenna slicing
            for bl in xrange(0, self.nbl, self.bl_diff):
                bl_end = min(bl + self.bl_diff, self.nbl)
                bl_diff = bl_end - bl
                cpu_slice['nbl'] = slice(bl,  bl_end, 1)
                gpu_slice['nbl'] = slice(0, bl_diff, 1)
                gpu_count['nbl'] = bl_diff

                # If we have one baseline, create
                # slices for the two related baselines,
                # obtained from the antenna pairs array
                # The antenna indices will be random
                # on the CPU side, but fit into indices
                # 0 and 1 on the GPU. We have the 'na1'
                # key, so that transfer_data can handle
                # the first and second antenna index
                if bl_diff == 1:
                    ant0 = self.ant_pairs_cpu[0, t, bl]
                    ant1 = self.ant_pairs_cpu[1, t, bl]
                    cpu_slice['na'] = slice(ant0, ant0 + 1, 1)
                    cpu_slice['na1'] = slice(ant1, ant1 + 1, 1)
                    gpu_slice['na'] = slice(0, 1, 1)
                    gpu_slice['na1'] = slice(1, 2, 1)
                    gpu_count['na'] = 2
                # Otherwise just take all antenna pairs
                # 'na1' will be ignored in this case
                elif bl_diff == self.nbl:
                    cpu_slice['na'] = slice(0, self.na, 1)
                    cpu_slice['na1'] = slice(0, self.na, 1)
                    gpu_slice['na'] = slice(0, self.na, 1)
                    gpu_slice['na1'] = slice(0, self.na, 1)
                    gpu_count['na'] = self.na
                else:
                    raise ValueError, ('Baseline difference (%s) '
                        'must be either 1 or the '
                        'total number of baselines (%s)') % \
                            (bl_diff, self.nbl)

                # Set up channel slicing
                for ch in xrange(0, self.nchan, self.chan_diff):
                    ch_end = min(ch + self.chan_diff, self.nchan)
                    ch_diff = ch_end - ch
                    cpu_slice['nchan'] = slice(ch, ch_end, 1)
                    gpu_slice['nchan'] = slice(0, ch_diff, 1)
                    gpu_count['nchan'] = ch_diff

                    # Set up source slicing
                    for src in xrange(0, self.nsrc, self.src_diff):
                        src_end = min(src + self.src_diff, self.nsrc)
                        src_diff = src_end - src
                        cpu_slice['nsrc'] = slice(src, src_end, 1)
                        gpu_slice['nsrc'] = slice(0, src_diff, 1)
                        gpu_count['nsrc'] = src_diff

                        # Set up the CPU source range slices
                        cpu_slice.update(mbu.source_range_slices(
                            src, src_end, src_nr_var_counts))

                        # and configure the same for GPU slices
                        for s in src_nr_vars:
                            cpu_var = cpu_slice[s]
                            gpu_slice[s] = slice(0, cpu_var.stop - cpu_var.start, 1)
                            gpu_count[s] = cpu_var.stop - cpu_var.start

                        yield (cpu_slice, gpu_slice, gpu_count)

    def __gen_sub_solvers(self):
        # Loop infinitely over the sub-solvers.
        while True:
            for i, subslvr in enumerate(self.solvers):
                yield (i, subslvr)

    def __validate_arrays(self, arrays):
        """
        Check that arrays are correctly configured
        """

        src_vars = mbu.source_nr_vars()
        vis_vars = ['ntime', 'nbl', 'na', 'nchan']

        for A in arrays:
            # Ensure they match ordering constraints
            order = [ORDERING_CONSTRAINTS[var]
                for var in A['shape'] if var in ORDERING_CONSTRAINTS]

            if not all([b >= a for a, b in zip(order, order[1:])]):
                raise ValueError(('Array %s does not follow '
                    'ordering constraints. Shape is %s, but '
                    'this does breaks the expecting ordering of %s ') % (
                        A['name'], A['shape'],
                        ORDERING_RANK))

            # Orthogonality of source variables and
            # time, baseline, antenna and channel
            nr_src_vars = [v for v in A['shape'] if v in src_vars]
            nr_vis_vars = [v for v in A['shape'] if v in vis_vars]

            if len(nr_src_vars) > 0 and len(nr_vis_vars) > 0:
                raise ValueError(('Array %s of shape %s '
                    'has source variables %s mixed with '
                    '%s. This solver does not currently '
                    'support this mix') % (
                        A['name'], A['shape'],
                        nr_src_vars, nr_vis_vars))

    def __twiddle_v4_arys_and_props(self, arys, props):
        # Add a custom transfer method for transferring
        # arrays to the sub-solver. Also, in general,
        # only maintain CPU arrays on the main solver,
        # but not GPU arrays, which will exist on the sub-solvers
        for ary in arys:
            ary['transfer_method'] = self.get_transfer_method(ary['name'])
            ary['gpu'] = False
            ary['cpu'] = True
            ary['aligned'] = True

        # Add custom property setter method
        for prop in props:
            prop['setter_method'] = self.get_setter_method(ary['name'])

        # Do not create CPU versions of result arrays
        for ary in [a for a in arys if a['name'] in
                ['vis', 'B_sqrt', 'jones', 'chi_sqrd_result']]:
            ary['cpu'] = False

        return arys, props

    def __twiddle_v4_subarys_and_props(self, arys, props):
        # Modify the array configuration for the sub-solvers
        # Don't create CPU arrays since we'll be copying them
        # from CPU arrays on the Composite Solver.
        # Do create GPU arrays, used for solving each sub-section
        # of the RIME.
        for ary in arys:
            # Add a transfer method
            ary['transfer_method'] = self.get_sub_transfer_method(ary['name'])
            ary['cpu'] = False
            ary['gpu'] = True

        return arys, props

    def __transfer_slice(self, r, cpu_ary, cpu_idx, gpu_ary, gpu_idx, stream):
        cpu_slice = cpu_ary[cpu_idx].squeeze()
        gpu_ary = gpu_ary[gpu_idx].squeeze()

        # If the slice is contiguous, pin the slice and copy it
        if cpu_slice.flags.c_contiguous is True:
            pinned_ary = cuda.register_host_memory(cpu_slice)
            copy_ary = pinned_ary
        # Pin the contiguous section containing the slice.
        # See if PyCUDA will handle the non-contiguous
        # slice copy internally (_memcpy_discontig)
        else:
            # Figure out the starting and ending indices
            start_idx = np.int32([0 if s.start is None else s.start
                for s in cpu_idx]) 
            end_idx = np.int32([s.stop
                if s.stop is not None else cpu_ary.shape[i]
                for i, s in enumerate(cpu_idx)]) 
            # 1D indices of the slice start and end
            # in the flattened array
            start = np.ravel_multi_index(start_idx, cpu_ary.shape)
            end = np.ravel_multi_index(end_idx-1, cpu_ary.shape)+1
            flat_slice = np.ravel(cpu_ary)[start:end]
            assert flat_slice.flags.c_contiguous is True
            # Pin the flat contiguous region
            pinned_ary = cuda.register_host_memory(flat_slice)
            # But copy the original slice
            copy_ary = cpu_slice

        #print 'Transferring %s with size %s shapes [%s vs %s]' % (
        #    r.name, mbu.fmt_bytes(copy_ary.nbytes), copy_ary.shape, gpu_ary.shape)

        gpu_ary.set_async(copy_ary, stream=stream)


    def transfer_arrays(self, sub_solver_idx,
        cpu_slice_map, gpu_slice_map):
        """
        Transfer CPU arrays on the CompositeBiroSolver over to the
        BIRO sub-solvers asynchronously.

        While it aims for generality, it generally depends on arrays
        having a ['nsrc', 'ntime', 'nbl'|'na', 'nchan'] ordering
        (its acceptable to have other dimensions between).
        Having mentioned this, arrays with 'nsrc' dimensions
        should generally not contain
        'ntime', 'nbl', 'na' or 'nchan' dimensions,
        IF you wish to transfer them from the CPU to the GPU.

        This is because slicing these dimensions may
        result in non-contiguous memory regions for transfer,
        and PyCUDA can only handle 2 degrees of
        of non-contiguity in arrays. If this function falls over
        due to non-contiguity, you now know why.

        The 'stokes' and 'alpha' arrays in v4 are examples
        where this principle is broken, but transfer still works
        with PyCUDA's non-contiguity handling as there is
        only one degree of it.

        The jones array is an example of a GPU only array that
        is not affected since it is never transferred.
        """
        i = sub_solver_idx
        subslvr = self.solvers[i]
        stream = self.stream[i]
        all_slice = slice(None,None,1)
        empty_slice = slice(0,0,1)

        two_ant_case = (subslvr.na == 2)

        for r in self.arrays.itervalues():
            # Is there anything to transfer for this array?
            if not r.cpu:
                #print '%s has no CPU array' % r.name
                continue

            cpu_name = mbu.cpu_name(r.name)
            gpu_name = mbu.gpu_name(r.name)

            # Get the CPU array on the composite solver
            # and the CPU array and the GPU array
            # on the sub-solver
            cpu_ary = getattr(self,cpu_name)
            gpu_ary = getattr(subslvr,gpu_name)

            if cpu_ary is None or gpu_ary is None:
                #print 'Skipping %s' % r.name
                continue
            else:
                #print 'Handling %s' % r.name
                pass

            # Set up the slicing of the main CPU array.
            # Map dimensions in cpu_slice_map
            # to the slice arguments, otherwise,
            # just take everything in the dimension
            cpu_idx = [cpu_slice_map[s]
                if s in cpu_slice_map else all_slice
                for s in r.sshape]

            gpu_idx = [gpu_slice_map[s]
                if s in gpu_slice_map else all_slice
                for s in r.sshape]

            # Bail if there's an empty slice in the index
            if gpu_idx.count(empty_slice) > 0:
                #print '%s has an empty slice, skipping' % r.name
                continue

            # Checking if we're handling two antenna here
            # A precursor to the vile hackery that follows
            try:
                na_idx = r.sshape.index('na')
            except ValueError:
                na_idx = -1

            # If we've got the two antenna case, slice
            # the gpu array at the first antenna position
            if two_ant_case and na_idx > 0:
                gpu_idx[na_idx] = 0

            self.__transfer_slice(r, cpu_ary, cpu_idx,
                gpu_ary, tuple(gpu_idx), stream)

            # Right, handle transfer of the second antenna's data
            if two_ant_case and na_idx > 0:
                # Slice the CPU and GPU arrays
                # at the second antenna position
                gpu_idx[na_idx] = 1
                cpu_idx[na_idx] = cpu_slice_map['na1']

                self.__transfer_slice(r, cpu_ary, cpu_idx,
                    gpu_ary, tuple(gpu_idx), stream)

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

        if not self.initialised:
            for i, subslvr in enumerate(self.solvers):
                subslvr.initialise()

            with self.solvers[0].context:
                # Get the reduction kernel
                # loaded in and hot
                pycuda.gpuarray.sum(
                    self.solvers[0].chi_sqrd_result_gpu,
                    stream=self.stream[0])

            self.initialised = True

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        nr_var_counts = mbu.sources_to_nr_vars(self.slvr_cfg[Options.SOURCES])
        subslvr_gen = self.__gen_sub_solvers()

        for cpu_slice_map, gpu_slice_map, gpu_count in self.__gen_rime_slices():
            i, subslvr = subslvr_gen.next()

            """"
            t = cpu_slice_map['ntime']
            bl = cpu_slice_map['nbl']
            ch = cpu_slice_map['nchan']
            src = cpu_slice_map['nsrc']

            print '%s: %s - %s/%s %s: %s - %s/%s %s: %s - %s/%s %s: %s - %s/%s' % (
                'T', t.start, t.stop, self.ntime,
                'BL', bl.start, bl.stop, self.nbl,
                'CH', ch.start, ch.stop, self.nchan,
                'SRC', src.start, src.stop, self.nsrc)
            """

            with subslvr.context:
                # Configure the number variable counts
                # on the sub solver
                subslvr.cfg_sub_dims(gpu_count)

                # Transfer arrays
                self.transfer_arrays(i, cpu_slice_map, gpu_slice_map)

                # Execute the kernels
                subslvr.rime_e_beam.execute(subslvr, self.stream[i])
                subslvr.rime_b_sqrt.execute(subslvr, self.stream[i])
                subslvr.rime_ekb_sqrt.execute(subslvr, self.stream[i])
                subslvr.rime_sum.execute(subslvr, self.stream[i])

    def shutdown(self):
        """ Shutdown the solver """
        with self.context as ctx:
            for slvr in self.solvers:
                slvr.shutdown()

    def get_properties(self):
        # Obtain base solver property dictionary
        # and add the beam cube dimensions to it
        D = super(CompositeBiroSolver, self).get_properties()

        D.update({
            Options.E_BEAM_WIDTH : self.beam_lw,
            Options.E_BEAM_HEIGHT : self.beam_mh,
            Options.E_BEAM_DEPTH : self.beam_nud
        })

        return D

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
