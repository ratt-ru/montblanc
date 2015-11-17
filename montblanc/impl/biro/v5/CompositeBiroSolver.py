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
import functools
import numpy as np
import types

import concurrent.futures as cf
import threading

import pycuda.driver as cuda
import pycuda.tools

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

        # Create thread local storage
        self.thread_local = threading.local()

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

        nsolvers = slvr_cfg.get('nsolvers', 2)
        self.dev_ctxs = slvr_cfg.get(Options.CONTEXT)

        self.__validate_arrays(A_sub)

        # Massage the contexts for each device into a list
        if not isinstance(self.dev_ctxs, list):
            self.dev_ctxs = [self.dev_ctxs]

        montblanc.log.info('Using %d solver(s) per device', nsolvers)

        # Shorten the type name
        C = CompositeBiroSolver

        # Create a one thread executor for each device context,
        # i.e. a thread per device
        executors = [cf.ThreadPoolExecutor(1) for ctx in self.dev_ctxs]

        for ex, ctx in zip(executors, self.dev_ctxs):
            ex.submit(C.__thread_init, self, ctx).result()

        # Find the budget with the lowest memory usage
        # Work with the device with the lowest memory
        budgets = sorted([ex.submit(C.__thread_budget, self,
                            slvr_cfg, A_sub, props).result()
                        for ex in executors],
                    key=lambda T: T[1])

        P, M, mem = budgets[0]

        # Log some information about the memory budget
        # and dimension reduction
        changes = ['%s: %s => %s' % (k, props[k], v)
            for k, v in M.iteritems()]

        montblanc.log.info(('Selecting a solver memory budget of %s '
            'for %d solvers. The following dimension '
            'reductions have been applied: %s.'),
                mbu.fmt_bytes(mem), nsolvers, ', '.join(changes))

        # Create the sub solver configuration
        subslvr_cfg = BiroSolverConfiguration(**slvr_cfg)
        subslvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_DEFAULTS
        subslvr_cfg[Options.NTIME] = P[Options.NTIME]
        subslvr_cfg[Options.NA] = P[Options.NA]
        subslvr_cfg[Options.NBL] = P[Options.NBL]
        subslvr_cfg[Options.NCHAN] = P[Options.NCHAN]
        subslvr_cfg[Options.CONTEXT] = ctx

        # Extract the dimension differences
        self.src_diff = P['nsrc']
        self.time_diff = P[Options.NTIME]
        self.ant_diff = P[Options.NA]
        self.bl_diff = P[Options.NBL]
        self.chan_diff = P[Options.NCHAN]

        # Now create the solvers on each thread
        for ex in executors:
            ex.submit(C.__thread_create_solvers,
                self, subslvr_cfg, P, nsolvers).result()

        A_sub, P_sub = self.__twiddle_v4_subarys_and_props(A_sub, P_sub)

        # Register arrays and properties on each thread's solvers
        for ex in executors:
            ex.submit(C.__thread_reg_sub_arys_and_props,
                self, A_sub, P_sub).result()

        self.executors = executors
        self.initialised = False

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
                else:
                    cpu_slice['na'] = slice(0, self.na, 1)
                    cpu_slice['na1'] = slice(0, self.na, 1)
                    gpu_slice['na'] = slice(0, self.na, 1)
                    gpu_slice['na1'] = slice(0, self.na, 1)
                    gpu_count['na'] = self.na

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

                        yield (cpu_slice.copy(), gpu_slice.copy(), gpu_count.copy())

    def __thread_gen_sub_solvers(self):
        # Loop infinitely over the sub-solvers.
        while True:
            for i, subslvr in enumerate(self.thread_local.solvers):
                yield (i, subslvr)

    def __gen_executors(self):
        # Loop indefinitely over executors
        first = True

        while True:
            for i, ex in enumerate(self.executors):
                yield(i, first, ex)

            first = False

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
            ary['transfer_method'] = self.__get_transfer_method(ary['name'])
            ary['gpu'] = False
            ary['cpu'] = True

        # Add custom property setter method
        for prop in props:
            prop['setter_method'] = self.__get_setter_method(ary['name'])

        # Do not create CPU versions of scratch arrays
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
            ary['transfer_method'] = self.__get_sub_transfer_method(ary['name'])
            ary['cpu'] = False
            ary['gpu'] = True
            # Don't initialise arrays on the sub-solvers,
            # it'll all get copied on anyway
            ary['default'] = None
            ary['test'] = None

        # We'll use memory pools for the X2 values
        for ary in [a for a in arys if a['name'] in ['X2']]:
            ary['cpu'] = False
            ary['gpu'] = False

        return arys, props

    def __transfer_slice(self, r, subslvr,
        cpu_ary, cpu_idx, gpu_ary, gpu_idx):
        cpu_slice = cpu_ary[cpu_idx].squeeze()
        gpu_ary = gpu_ary[gpu_idx].squeeze()

        # Obtain some pinned memory from the memory pool
        staged_ary = subslvr.pinned_mem_pool.allocate(
            shape=gpu_ary.shape, dtype=gpu_ary.dtype)

        # Copy data into staging area
        staged_ary[:] = cpu_slice

        #montblanc.log.info('Transferring %s with size %s shapes [%s vs %s]',
        #    r.name, mbu.fmt_bytes(staged_ary.nbytes), staged_ary.shape, gpu_ary.shape)

        gpu_ary.set_async(staged_ary, stream=subslvr.stream)

    def __transfer_arrays(self, sub_solver_idx,
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
        subslvr = self.thread_local.solvers[i]
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

            # For the one baseline, two antenna case,
            # hard code the antenna indices on the GPU
            # to 0 and 1
            if two_ant_case and r.name == 'ant_pairs':
                cpu_idx = [all_slice for s in r.shape]
                cpu_ary = np.array([0,1]).reshape(subslvr.ant_pairs_shape)

            self.__transfer_slice(r, subslvr,
                cpu_ary, cpu_idx,
                gpu_ary, tuple(gpu_idx))

            # Right, handle transfer of the second antenna's data
            if two_ant_case and na_idx > 0:
                # Slice the CPU and GPU arrays
                # at the second antenna position
                gpu_idx[na_idx] = 1
                cpu_idx[na_idx] = cpu_slice_map['na1']

                self.__transfer_slice(r, subslvr,
                    cpu_ary, cpu_idx,
                    gpu_ary, tuple(gpu_idx))

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

    def __thread_init(self, context):
        """
        Initialise the current thread, by associating
        a CUDA context with it, and pushing the context.
        """
        montblanc.log.debug('Pushing CUDA context in thread %s',
            threading.current_thread())
        context.push()
        self.thread_local.context = context

    def __thread_shutdown(self):
        """
        Shutdown the current thread,
        by popping the associated CUDA context
        """
        montblanc.log.debug('Popping CUDA context in thread %s',
            threading.current_thread())
        self.thread_local.context.pop()

    def __thread_budget(self, slvr_cfg, A_sub, props):
        """
        Get memory budget and dimension reduction
        information from the CUDA device associated
        with the current thread and context
        """
        montblanc.log.debug('Budgeting in thread %s', threading.current_thread())

        # Query free memory on this context
        (free_mem,total_mem) = cuda.mem_get_info()

        # Work with a supplied memory budget, otherwise use
        # free memory less an amount equal to the upper size
        # of an NVIDIA context
        mem_budget = slvr_cfg.get('mem_budget', free_mem - 100*ONE_MB)

        nsolvers = slvr_cfg.get('nsolvers', 2)
        na = slvr_cfg.get(Options.NA)
        nsrc = 400
        src_str_list = ['nsrc'] + mbu.source_nr_vars()
        src_reduction_str = '&'.join(['%s=%s' % (nr_var, nsrc)
            for nr_var in src_str_list])

        # Figure out a viable dimension configuration
        # given the total problem size 
        viable, modded_dims = mbu.viable_dim_config(
            mem_budget, A_sub, props,
                [src_reduction_str, 'ntime',
                'nbl=%s&na=%s' % (na, na)
                ,'nbl=1&na=2',
                'nchan=50%'],
            nsolvers)                

        # Create property dictionary with updated
        # dimensions.
        P = props.copy()
        P.update(modded_dims)

        required_mem = mbu.dict_array_bytes_required(A_sub, P)

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

        return P, modded_dims, required_mem

    def __thread_create_solvers(self, subslvr_cfg, P, nsolvers):
        """
        Create solvers on the thread local data
        """
        montblanc.log.debug('Creating solvers in thread %s',
            threading.current_thread())

        # Pre-allocate a 16KB GPU memory pool
        # for each device, this is needed to
        # prevent the PyCUDA reduction functions
        # allocating memory and stalling the
        # asynchronous pipeline.
        dev_mem_pool = pycuda.tools.DeviceMemoryPool()
        dev_mem_pool.allocate(16*ONE_KB).free()

        # Pre-allocate a 16KB pinned memory pool
        # This is used to hold the results of PyCUDA
        # reduction kernels.
        pinned_mem_pool = pycuda.tools.PageLockedMemoryPool()
        pinned_mem_pool.allocate(shape=(16*ONE_KB,),
            dtype=np.int8).base.free()

        # Configure thread local storage
        # Number of solvers in this thread
        self.thread_local.nsolvers = nsolvers
        # List of solvers used by this thread
        self.thread_local.solvers = [False for s in range(nsolvers)]
        # Has there been a previous iteration on this solver
        self.thread_local.prev_iteration = [False for s in range(nsolvers)]
        # Initialise the X2 sum to zero
        self.thread_local.X2 = self.ft(0.0)
        # Initialise the subsolver generator
        self.thread_local.subslvr_gen = self.__thread_gen_sub_solvers()

        # Set the CUDA context in the configuration to
        # the one associated with this thread
        subslvr_cfg[Options.CONTEXT] = self.thread_local.context

        # Create solvers for this context
        for i, s in enumerate(range(nsolvers)):
            subslvr = BiroSolver(subslvr_cfg)
            # Configure the total number of sources
            # handled by each sub-solver
            subslvr.cfg_total_src_dims(P['nsrc'])
            subslvr.set_dev_mem_pool(dev_mem_pool)
            subslvr.set_pinned_mem_pool(pinned_mem_pool)
            self.thread_local.solvers[i] = subslvr

    def __thread_reg_sub_arys_and_props(self, A_sub, P_sub):
        """
        Register arrays and properties on
        the thread local solvers
        """
        montblanc.log.debug('Registering arrays and properties in thread %s',
            threading.current_thread())
        # Create the arrays on the sub solvers
        for i, subslvr in enumerate(self.thread_local.solvers):
            subslvr.register_properties(P_sub)
            subslvr.register_arrays(A_sub)

    def __thread_solve_sub(self, cpu_slice_map, gpu_slice_map, gpu_count, first=False):
        """
        Solve a portion of the RIME, specified by the cpu_slice_map and
        gpu_slice_map dictionaries.
        """
        tl = self.thread_local

        # If this is flagged as the first iteration, reset variables
        if first is True:
            # There has been no previous iteration on this solver
            tl.prev_iteration = [False for s in range(tl.nsolvers)]
            # Initialise the X2 sum to zero
            tl.X2 = self.ft(0.0)
            # Initialise the subsolver generator
            tl.subslvr_gen = self.__thread_gen_sub_solvers()

        i, subslvr = tl.subslvr_gen.next()

        # If the solver iterated previously, there's
        # a X2 that needs to be extracted
        if tl.prev_iteration[i]:
            # Get an array from the pinned memory pool
            sub_X2 = subslvr.pinned_mem_pool.allocate(
                shape=self.X2_shape, dtype=self.X2_dtype)
            
            # Copy the X2 value off the GPU onto the CPU
            subslvr.rime_reduce.X2_gpu_ary.get_async(
                ary=sub_X2, stream=subslvr.stream)

            # Synchronise before extracting the X2 value
            subslvr.stream.synchronize()

            # Add to the running total on the local thread
            tl.X2 += sub_X2

        # Configure the number variable counts
        # on the sub solver
        subslvr.cfg_sub_dims(gpu_count)

        # Transfer arrays
        self.__transfer_arrays(i, cpu_slice_map, gpu_slice_map)

        # Pre-execution (async copy constant data to the GPU)
        subslvr.rime_e_beam.pre_execution(subslvr, subslvr.stream)
        subslvr.rime_b_sqrt.pre_execution(subslvr, subslvr.stream)
        subslvr.rime_ekb_sqrt.pre_execution(subslvr, subslvr.stream)
        subslvr.rime_sum.pre_execution(subslvr, subslvr.stream)
        subslvr.rime_reduce.pre_execution(subslvr, subslvr.stream)

        prev_i = (i-1) % len(tl.prev_iteration)

        # Wait for previous kernel execution to finish
        # on the previous solver (stream) before launching new
        # kernels on the current stream
        if tl.prev_iteration[prev_i]:
            prev_slvr = tl.solvers[prev_i]
            subslvr.stream.wait_for_event(prev_slvr.kernels_done)

        # Execute the kernels
        subslvr.rime_e_beam.execute(subslvr, subslvr.stream)
        subslvr.rime_b_sqrt.execute(subslvr, subslvr.stream)
        subslvr.rime_ekb_sqrt.execute(subslvr, subslvr.stream)
        subslvr.rime_sum.execute(subslvr, subslvr.stream)
        subslvr.rime_reduce.execute(subslvr, subslvr.stream)

        # Record kernel completion
        subslvr.kernels_done.record(subslvr.stream)

        # Indicate that this solver has executed and
        # that a X2 is waiting for extraction
        tl.prev_iteration[i] = True

    def __thread_solve_sub_final(self):
        """
        Retrieve any final X2 values from the solvers and
        return the total X2 sum for this thread.
        """
        montblanc.log.debug('Retrieve final X2 in thread %s', threading.current_thread())
        tl = self.thread_local

        # Retrieve final X2 values
        for j in range(tl.nsolvers):
            i, subslvr = tl.subslvr_gen.next()

            # If the solver hasn't executed, no X2 is available
            if not tl.prev_iteration[i]:
                continue

            # Get an array from the pinned memory pool
            sub_X2 = subslvr.pinned_mem_pool.allocate(
                shape=self.X2_shape, dtype=self.X2_dtype)
            
            # Copy the X2 value off the GPU onto the CPU
            subslvr.rime_reduce.X2_gpu_ary.get_async(
                ary=sub_X2, stream=subslvr.stream)

            # Synchronise before extracting the X2 value
            subslvr.stream.synchronize()

            tl.X2 += sub_X2

        return tl.X2

    def initialise(self):
        """ Initialise the sub-solver """

        def __init_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.initialise()

        if not self.initialised:
            for ex in self.executors:
                ex.submit(__init_func).result()

            self.initialised = True

    @staticmethod
    def __rm_future_cb(f, Q):
        # There's no actual result, but asking for it
        # will throw any exceptions from the future execution
        f.result()
        Q.remove(f)

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        # For easier typing
        C = CompositeBiroSolver
        # Is this the first iteration of this executor?
        first = [True for ex in self.executors]
        # Queue of futures for each executor
        future_Q = [[] for ex in self.executors]
        # Callbacks for removing futures from the above Q,
        # for each executor
        nr_ex = len(self.executors)
        rm_fut_cb = [functools.partial(C.__rm_future_cb, Q=future_Q[i])
            for i in range(nr_ex)]

        # Iterate over the RIME space, i.e. slices over the CPU and GPU
        for cpu_slice_map, gpu_slice_map, gpu_count in self.__gen_rime_slices():
            # Attempt to submit work to an executor
            submitted = False

            while not submitted:
                for i, ex in enumerate(self.executors):
                    # Try another executor if there's too much work on this queue
                    if len(future_Q[i]) > 2:
                        continue

                    # Submit work to the thread, solve this portion of the RIME
                    f = ex.submit(C.__thread_solve_sub, self,
                        cpu_slice_map, gpu_slice_map, gpu_count, first=first[i])

                    # Add the future to the queue
                    future_Q[i].append(f)

                    # Add a callback removing the future from the appropriate queue
                    # once it completes                    
                    f.add_done_callback(rm_fut_cb[i])

                    # This section of work has been submitted,
                    # break out of the for loop
                    submitted = True
                    first[i] = False
                    break

                # OK, all our executors are really busy,
                # wait for one of their oldest tasks to finish
                if not submitted:
                    try:
                        wait_f = [future_Q[i][0] for i in range(len(future_Q))]
                        cf.wait(wait_f, return_when=cf.FIRST_COMPLETED)
                    # This case happens when future_Q[i][0] is attempted,
                    # but the future callback has removed it from the queue
                    # Have another go at the executors
                    except IndexError as e:
                        pass

        # For each executor (thread), request the final X2 result
        # as a future, sum them together to produce the final X2
        self.X2_cpu = np.sum([ex.submit(C.__thread_solve_sub_final, self).result() for
            ex in self.executors]).reshape(self.X2_shape)

    def shutdown(self):
        """ Shutdown the solver """
        def __shutdown_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.shutdown()

            self.__thread_shutdown()

        for ex in self.executors:
            ex.submit(__shutdown_func).result()

        self.initialised = False

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

    def __get_setter_method(self,name):
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

    def __get_sub_transfer_method(self,name):
        def f(self, npary):
            raise Exception, 'Its illegal to call set methods on the sub-solvers'
        return types.MethodType(f,self)

    def __get_transfer_method(self, name):
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
