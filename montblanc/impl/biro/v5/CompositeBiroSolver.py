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

import functools
import itertools
import numpy as np
import types
import sys

import concurrent.futures as cf
import threading

import pycuda.driver as cuda
import pycuda.tools

import montblanc
import montblanc.util as mbu

from montblanc.solvers import MontblancNumpySolver
from montblanc.config import BiroSolverConfig as Options

from montblanc.impl.biro.v4.config import (
    A as v4Arrays,
    P as v4Props,
    Classifier)

from hypercube.dims import DIMDATA

import montblanc.impl.biro.v4.BiroSolver as BSV4mod

from montblanc.impl.biro.v5.BiroSolver import BiroSolver

NA_EXTRA = 'na1'

ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

ALL_SLICE = slice(None,None,1)
EMPTY_SLICE = slice(0,0,1)

ORDERING_CONSTRAINTS = { nr_var : 1 for nr_var in mbu.source_nr_vars() }
ORDERING_CONSTRAINTS.update({ 'nsrc' : 1,
    'ntime': 2, 'nbl': 3, 'na': 3, 'nchan': 4 })

ORDERING_RANK = [' or '.join(['nsrc'] + mbu.source_nr_vars()),
    'ntime', ' or '.join(['nbl', 'na']), 'nchan']

class CompositeBiroSolver(MontblancNumpySolver):
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
        super(CompositeBiroSolver, self).__init__(slvr_cfg=slvr_cfg)

        # Create thread local storage
        self.thread_local = threading.local()

        self.register_default_dimensions()

        # Configure the dimensions of the beam cube
        self.register_dimension('beam_lw',
            slvr_cfg[Options.E_BEAM_WIDTH],
            description='E cube l width')

        self.register_dimension('beam_mh',
            slvr_cfg[Options.E_BEAM_HEIGHT],
            description='E cube m height')

        self.register_dimension('beam_nud',
            slvr_cfg[Options.E_BEAM_DEPTH],
            description='E cube nu depth')

        # Monkey patch v4 antenna pair functions into the object
        from montblanc.impl.biro.v4.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        # Copy the v4 arrays and properties and
        # modify them for use on this Composite Solver
        A_main, P_main = self._cfg_comp_slvr_arys_and_props(v4Arrays, v4Props)

        self.register_properties(P_main)
        self.register_arrays(A_main)

        # PyCUDA contexts for each GPU device   
        self.dev_ctxs = slvr_cfg.get(Options.CONTEXT)
        # Number of GPU Solvers created for each device
        nsolvers = slvr_cfg.get(Options.NSOLVERS)
        # Maximum number of enqueued visibility chunks
        # before throttling is applied
        self.throttle_factor = slvr_cfg.get(
            Options.VISIBILITY_THROTTLE_FACTOR)

        # Massage the contexts for each device into a list
        if not isinstance(self.dev_ctxs, list):
            self.dev_ctxs = [self.dev_ctxs]

        montblanc.log.info('Using {d} solver(s) per device.'.format(
            d=nsolvers))

        # Shorten the type name
        C = CompositeBiroSolver

        # Create a one thread executor for each device context,
        # i.e. a thread per device
        executors = [cf.ThreadPoolExecutor(1) for ctx in self.dev_ctxs]
        sync_executors = [cf.ThreadPoolExecutor(1) for ex in executors]

        montblanc.log.info('Created {d} executor(s).'.format(d=len(executors)))

        for ex, ctx in zip(executors, self.dev_ctxs):
            ex.submit(C._thread_init, self, ctx).result()

        for ex, ctx in zip(sync_executors, self.dev_ctxs):
            ex.submit(C._thread_init, self, ctx).result()

        montblanc.log.info('Initialised {d} thread(s).'.format(d=len(executors)))

        # Get a template dictionary
        T = self.template_dict()

        A_sub, P_sub = self._cfg_sub_slvr_arys_and_props(v4Arrays, v4Props)
        self._validate_arrays(A_sub)

        # Find the budget with the lowest memory usage
        # Work with the device with the lowest memory
        budgets = sorted([ex.submit(C._thread_budget, self,
                            slvr_cfg, A_sub, T).result()
                        for ex in executors],
                    key=lambda T: T[1])

        P, M, mem = budgets[0]

        # Log some information about the memory budget
        # and dimension reduction
        montblanc.log.info(('Selected a solver memory budget of {b} '
            'for {d} solvers.').format(b=mbu.fmt_bytes(mem), d=nsolvers))

        montblanc.log.info(('The following dimension reductions '
            'have been applied:'))

        for k, v in M.iteritems():
            montblanc.log.info('{p}{d}: {id} => {rd}'.format
                (p=' '*4, d=k, id=T[k], rd=v))

        # Create the sub solver configuration
        subslvr_cfg = slvr_cfg.copy()
        subslvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_EMPTY
        subslvr_cfg[Options.CONTEXT] = ctx
        subslvr_cfg[Options.SOLVER_TYPE] = Options.SOLVER_TYPE_SLAVE

        subslvr_cfg = self._cfg_subslvr_dims(subslvr_cfg, P)

        # Extract the dimension differences
        self.src_diff = P[Options.NSRC]
        self.time_diff = P[Options.NTIME]
        self.ant_diff = P[Options.NA]
        self.bl_diff = P[Options.NBL]
        self.chan_diff = P[Options.NCHAN]

        montblanc.log.info('Creating {s} solver(s) on {d} device(s).'
            .format(s=nsolvers, d=len(executors)))

        # Now create the solvers on each thread
        for ex in executors:
            ex.submit(C._thread_create_solvers,
                self, subslvr_cfg, P, nsolvers).result()

        montblanc.log.info('Solvers Created')

        # Register arrays and properties on each thread's solvers
        for ex in executors:
            ex.submit(C._thread_reg_sub_arys_and_props,
                self, A_sub, P_sub).result()

        self.executors = executors
        self.sync_executors = sync_executors
        self.initialised = False

    def _gen_source_slices(self):
        """
        Iterate over the visibility space in chunks, returning a
        dictionary of slices keyed on the following dimensions:
            nsrc, npsrc, ngsrc, nssrc, ...
        """
        src_nr_var_counts = mbu.sources_to_nr_vars(
            self._slvr_cfg[Options.SOURCES])
        src_nr_vars = mbu.source_nr_vars()

        nsrc = self.dim_local_size(Options.NSRC)

        # Create the slice dictionaries, which we use to index
        # dimensions of the CPU and GPU array.
        cpu_slice, gpu_slice = {}, {}

        # Set up source slicing
        for src in xrange(0, nsrc, self.src_diff):
            src_end = min(src + self.src_diff, nsrc)
            src_diff = src_end - src
            cpu_slice[Options.NSRC] = slice(src, src_end, 1)
            gpu_slice[Options.NSRC] = slice(0, src_diff, 1)

            # Set up the CPU source range slices
            cpu_slice.update(mbu.source_range_slices(
                src, src_end, src_nr_var_counts))

            # and configure the same for GPU slices
            for s in src_nr_vars:
                cpu_var = cpu_slice[s]
                gpu_slice[s] = slice(0, cpu_var.stop - cpu_var.start, 1)

            yield (cpu_slice.copy(), gpu_slice.copy())

    def _gen_vis_slices(self):
        """
        Iterate over the visibility space in chunks, returning a
        dictionary of slices keyed on the following dimensions:
            ntime, nbl, na, na1, nchan
        """

        ntime, nbl, na, nchan = self.dim_local_size(
            'ntime', 'nbl', 'na', 'nchan')

        # Create the slice dictionaries, which we use to index
        # dimensions of the CPU and GPU array.
        cpu_slice, gpu_slice = {}, {}

        montblanc.log.info('Generating RIME slices')

        # Set up time slicing
        for t in xrange(0, ntime, self.time_diff):
            t_end = min(t + self.time_diff, ntime)
            t_diff = t_end - t
            cpu_slice[Options.NTIME] = slice(t,  t_end, 1)
            gpu_slice[Options.NTIME] = slice(0, t_diff, 1)

            # Set up baseline and antenna slicing
            for bl in xrange(0, nbl, self.bl_diff):
                bl_end = min(bl + self.bl_diff, nbl)
                bl_diff = bl_end - bl
                cpu_slice[Options.NBL] = slice(bl,  bl_end, 1)
                gpu_slice[Options.NBL] = slice(0, bl_diff, 1)

                # If we have one baseline, create
                # slices for the two related baselines,
                # obtained from the antenna pairs array
                # The antenna indices will be random
                # on the CPU side, but fit into indices
                # 0 and 1 on the GPU. We have the NA_EXTRA
                # key, so that transfer_data can handle
                # the first and second antenna index
                if bl_diff == 1:
                    ant0 = self.ant_pairs[0, t, bl]
                    ant1 = self.ant_pairs[1, t, bl]
                    cpu_slice[Options.NA] = slice(ant0, ant0 + 1, 1)
                    cpu_slice[NA_EXTRA] = slice(ant1, ant1 + 1, 1)
                    gpu_slice[Options.NA] = slice(0, 1, 1)
                    gpu_slice[NA_EXTRA] = slice(1, 2, 1)
                # Otherwise just take all antenna pairs
                # NA_EXTRA will be ignored in this case
                else:
                    cpu_slice[Options.NA] = slice(0, na, 1)
                    cpu_slice[NA_EXTRA] = slice(0, na, 1)
                    gpu_slice[Options.NA] = slice(0, na, 1)
                    gpu_slice[NA_EXTRA] = slice(0, na, 1)

                # Set up channel slicing
                for ch in xrange(0, nchan, self.chan_diff):
                    ch_end = min(ch + self.chan_diff, nchan)
                    ch_diff = ch_end - ch
                    cpu_slice[Options.NCHAN] = slice(ch, ch_end, 1)
                    gpu_slice[Options.NCHAN] = slice(0, ch_diff, 1)

                    yield (cpu_slice.copy(), gpu_slice.copy())

    def _thread_gen_sub_solvers(self):
        # Loop infinitely over the sub-solvers.
        while True:
            for i, subslvr in enumerate(self.thread_local.solvers):
                yield (i, subslvr)

    def _validate_arrays(self, arrays):
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

    def _cfg_subslvr_dims(self, subslvr_cfg, P):
        for dim in self._dims.itervalues():
            name = dim[DIMDATA.NAME]
            if name in P:
                # Copy dimension data for reconfiguration
                sub_dim = dim.copy()

                sub_dim.update({
                    DIMDATA.LOCAL_SIZE: P[name],
                    DIMDATA.EXTENTS: [0, P[name]],
                    DIMDATA.SAFETY: False })

                subslvr_cfg[name] = sub_dim

        return subslvr_cfg


    def _cfg_comp_slvr_arys_and_props(self, arys, props):
        """
        Configure arrays and properties for the main
        Composite Solver (self)
        """
        
        # Don't store intermediate arrays
        arys = [a.copy() for a in arys if
            Classifier.GPU_SCRATCH not in a['classifiers']]

        # Copy properties
        props = [p.copy() for p in props]

        # Add custom property setter method
        for prop in props:
            prop['setter_method'] = self._get_setter_method(prop['name'])

        return arys, props

    def _cfg_sub_slvr_arys_and_props(self, arys, props):
        """
        Modify the array configuration for the sub-solvers
        Add transfer methods and null out any default and test
        data sources since we'll expect this solver to initialise them.
        """

        arys = [a.copy() for a in arys]
        props = [p.copy() for p in props]

        for ary in arys:
            # Add a transfer method
            ary['transfer_method'] = self._get_sub_transfer_method(ary['name'])
            # Don't initialise arrays on the sub-solvers,
            # it'll all get copied on anyway
            ary['default'] = None
            ary['test'] = None

        return arys, props

    def _enqueue_const_data_htod(self, subslvr, device_ptr):
        """
        Enqueue an async copy of the constant data array
        from the sub-solver into the constant memory buffer
        referenced by the device pointer.
        """
        # Get sub solver constant data array
        host_ary = subslvr.const_data().ndary()

        # Allocate pinned memory with same size
        pinned_ary = subslvr.pinned_mem_pool.allocate(
            shape=host_ary.shape, dtype=host_ary.dtype)

        # Copy into pinned memory
        pinned_ary[:] = host_ary

        # Enqueue the asynchronous transfer
        cuda.memcpy_htod_async(device_ptr, pinned_ary,
            stream=subslvr.stream)

        return pinned_ary

    def _enqueue_array_slice_htod(self, r, subslvr,
        cpu_ary, cpu_idx, gpu_ary, gpu_idx):
        """
        Copies a slice of the CPU array into a slice of the GPU array.
        """

        cpu_slice = cpu_ary[cpu_idx].squeeze()
        gpu_ary = gpu_ary[gpu_idx].squeeze()

        # Obtain some pinned memory from the memory pool
        pinned_ary = subslvr.pinned_mem_pool.allocate(
            shape=gpu_ary.shape, dtype=gpu_ary.dtype)

        # Copy data into pinned memory
        pinned_ary[:] = cpu_slice

        #montblanc.log.info('Transferring %s with size %s shapes [%s vs %s]',
        #    r.name, mbu.fmt_bytes(staged_ary.nbytes), staged_ary.shape, gpu_ary.shape)

        gpu_ary.set_async(pinned_ary, stream=subslvr.stream)

        return pinned_ary

    def _enqueue_array_htod(self, subslvr,
        cpu_slice_map, gpu_slice_map, classifiers=None):
        """
        Enqueue asynchronous copies from CPU arrays on the
        CompositeBiroSolver to GPU arrays on the BIRO sub-solvers
        on the CUDA stream associated with a sub-solver.

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
        pool_refs = []
        
        na = subslvr.dim_local_size('na')
        two_ant_case = (na == 2)

        if classifiers is None:
            classifiers = frozenset()
        elif isinstance(classifiers, list):
            classifiers = frozenset(classifiers)
        elif isinstance(classifiers, Classifier):
            classifiers = frozenset((classifiers,))

        for r in self.arrays().itervalues():
            # Ignore this array if we don't get matching classifiers
            if len(classifiers.intersection(r['classifiers'])) == 0:
                continue

            # Get the CPU array on the composite solver
            # and the CPU array and the GPU array
            # on the sub-solver
            cpu_ary = getattr(self, r.name)
            gpu_ary = getattr(subslvr, r.name)

            # if cpu_ary is None or gpu_ary is None:
            #     continue
            # else:
            #     print 'Transferring {n}'.format(n=r.name)

            # Set up the slicing of the main CPU array.
            # Map dimensions in cpu_slice_map
            # to the slice arguments, otherwise,
            # just take everything in the dimension
            cpu_idx = [cpu_slice_map[s]
                if s in cpu_slice_map else ALL_SLICE
                for s in r.sshape]

            gpu_idx = [gpu_slice_map[s]
                if s in gpu_slice_map else ALL_SLICE
                for s in r.sshape]

            # Bail if there's an empty slice in the index
            if gpu_idx.count(EMPTY_SLICE) > 0:
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
                cpu_idx = [ALL_SLICE for s in r.shape]
                cpu_ary = np.array([0,1]).reshape(subslvr.ant_pairs_shape)

            pinned_ary = self._enqueue_array_slice_htod(r, subslvr,
                cpu_ary, cpu_idx,
                gpu_ary, tuple(gpu_idx))
            pool_refs.append(pinned_ary)

            # Right, handle transfer of the second antenna's data
            if two_ant_case and na_idx > 0:
                # Slice the CPU and GPU arrays
                # at the second antenna position
                gpu_idx[na_idx] = 1
                cpu_idx[na_idx] = cpu_slice_map[NA_EXTRA]

                pinned_ary = self._enqueue_array_slice_htod(r, subslvr,
                    cpu_ary, cpu_idx,
                    gpu_ary, tuple(gpu_idx))
                pool_refs.append(pinned_ary)

        return pool_refs

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

    def _thread_init(self, context):
        """
        Initialise the current thread, by associating
        a CUDA context with it, and pushing the context.
        """
        montblanc.log.debug('Pushing CUDA context in thread %s',
            threading.current_thread())
        context.push()
        self.thread_local.context = context

    def _thread_shutdown(self):
        """
        Shutdown the current thread,
        by popping the associated CUDA context
        """
        montblanc.log.debug('Popping CUDA context in thread %s',
            threading.current_thread())
        self.thread_local.context.pop()

    def _thread_budget(self, slvr_cfg, A_sub, props):
        """
        Get memory budget and dimension reduction
        information from the CUDA device associated
        with the current thread and context
        """
        montblanc.log.debug('Budgeting in thread %s', threading.current_thread())

        # Query free memory on this context
        (free_mem,total_mem) = cuda.mem_get_info()

        device = self.thread_local.context.get_device()

        montblanc.log.info('{d}: {t} total {f} free.'.format(
           d=device.name(), f=mbu.fmt_bytes(free_mem), t=mbu.fmt_bytes(total_mem)))

        # Work with a supplied memory budget, otherwise use
        # free memory less an amount equal to the upper size
        # of an NVIDIA context
        mem_budget = slvr_cfg.get('mem_budget', free_mem - 200*ONE_MB)

        nsolvers = slvr_cfg.get(Options.NSOLVERS)
        na = slvr_cfg.get(Options.NA)
        nsrc = slvr_cfg.get(Options.SOURCE_BATCH_SIZE)
        src_str_list = [Options.NSRC] + mbu.source_nr_vars()
        src_reduction_str = '&'.join(['%s=%s' % (nr_var, nsrc)
            for nr_var in src_str_list])

        ntime_split = 'ntime={n}'.format(n=100 // nsolvers)

        # Figure out a viable dimension configuration
        # given the total problem size 
        viable, modded_dims = mbu.viable_dim_config(
            mem_budget, A_sub, props,
                [ntime_split, src_reduction_str, 'ntime',
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

    def _thread_create_solvers(self, subslvr_cfg, P, nsolvers):
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
        # Initialise the subsolver generator
        self.thread_local.subslvr_gen = self._thread_gen_sub_solvers()

        # Set the CUDA context in the configuration to
        # the one associated with this thread
        subslvr_cfg[Options.CONTEXT] = self.thread_local.context

        # Create solvers for this context
        for i in range(nsolvers):
            subslvr = BiroSolver(subslvr_cfg)

            # Configure the source dimensions of each sub-solver.
            # Change the local size of each source dim so that there is
            # enough space in the associated arrays for NSRC sources.
            # Initially, configure the extents to be [0, NSRC], although
            # this will be setup properly in _thread_solve_sub
            nsrc = P[Options.NSRC]

            U = [{
                DIMDATA.NAME: nr_var,
                DIMDATA.LOCAL_SIZE: nsrc if nsrc < P[nr_var] else P[nr_var],
                DIMDATA.EXTENTS: [0, nsrc if nsrc < P[nr_var] else P[nr_var]],
                DIMDATA.SAFETY: False
            } for nr_var in [Options.NSRC] + mbu.source_nr_vars()]

            subslvr.update_dimensions(U)

            subslvr.set_dev_mem_pool(dev_mem_pool)
            subslvr.set_pinned_mem_pool(pinned_mem_pool)
            self.thread_local.solvers[i] = subslvr

    def _thread_reg_sub_arys_and_props(self, A_sub, P_sub):
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

    def _thread_enqueue_solve_batch(self, cpu_slice_map, gpu_slice_map, **kwargs):
        """
        Enqueue CUDA memory transfer and kernel execution operations on a CUDA stream.

        CPU and GPU arrays are sliced given the dimension:slice mapping specified in
        cpu_slice_map and gpu_slice_map.

        Returns a (event, X2) tuple, where event is a CUDA event recorded at the
        end of this sequence of operations and X2 is a pinned memory array that
        will hold the result of the chi-squared operation.
        """

        tl = self.thread_local
        i, subslvr = tl.subslvr_gen.next()

        # A list of references to memory pool allocated objects
        # ensuring that said objects remained allocated until
        # after compute has been performed. Returned from
        # this function, this object should be discarded when
        # reading the result of the enqueued operations.
        pool_refs = []

        # TODO: Classifer.X2_INPUT and Classifier.TELESCOPE_INPUT arrays
        # really only need to be transferred once for each visibility chunk

        # Now, iterate over our source chunks
        for src_cpu_slice_map, src_gpu_slice_map in self._gen_source_slices():
            # Update our maps with source slice information
            cpu_slice_map.update(src_cpu_slice_map)
            gpu_slice_map.update(src_gpu_slice_map)

            # Configure dimension extents on the sub-solver
            subslvr.update_dimensions([
                { DIMDATA.NAME: dim, DIMDATA.EXTENTS: [S.start, S.stop] }
                for dim, S in cpu_slice_map.iteritems() if dim != NA_EXTRA])

            # Enqueue E Beam
            kernel = subslvr.rime_e_beam
            pool_refs.extend(self._enqueue_array_htod(
                subslvr, cpu_slice_map, gpu_slice_map,
                classifiers=[Classifier.E_BEAM_INPUT]))
            pool_refs.append(self._enqueue_const_data_htod(
                subslvr, kernel.rime_const_data[0]))
            kernel.execute(subslvr, subslvr.stream)

            # Enqueue B Sqrt
            kernel = subslvr.rime_b_sqrt
            pool_refs.extend(self._enqueue_array_htod(
                subslvr, cpu_slice_map, gpu_slice_map,
                classifiers=[Classifier.B_SQRT_INPUT]))
            pool_refs.append(self._enqueue_const_data_htod(
                subslvr, kernel.rime_const_data[0]))
            kernel.execute(subslvr, subslvr.stream)

            # Enqueue EKB Sqrt
            kernel = subslvr.rime_ekb_sqrt
            pool_refs.extend(self._enqueue_array_htod(
                subslvr, cpu_slice_map, gpu_slice_map,
                classifiers=[Classifier.EKB_SQRT_INPUT]))
            pool_refs.append(self._enqueue_const_data_htod(
                subslvr, kernel.rime_const_data[0]))
            kernel.execute(subslvr, subslvr.stream)

            # Enqueue Sum Coherencies
            kernel = subslvr.rime_sum
            pool_refs.extend(self._enqueue_array_htod(
                subslvr, cpu_slice_map, gpu_slice_map,
                classifiers=[Classifier.COHERENCIES_INPUT]))
            pool_refs.append(self._enqueue_const_data_htod(
                subslvr, kernel.rime_const_data[0]))
            kernel.execute(subslvr, subslvr.stream)

        # Enqueue chi-squared term reduction and return the
        # GPU array allocated to it
        X2_gpu_ary = subslvr.rime_reduce.execute(subslvr, subslvr.stream)

        # Get pinned memory to hold the chi-squared result
        sub_X2 = subslvr.pinned_mem_pool.allocate(
            shape=self.X2.shape, dtype=self.X2.dtype)
        
        # Enqueue chi-squared copy off the GPU onto the CPU
        X2_gpu_ary.get_async(ary=sub_X2, stream=subslvr.stream)

        # Create and record an event directly after the chi-squared copy
        # We'll synchronise on this thread in our synchronisation executor
        sync_event = cuda.Event(cuda.event_flags.DISABLE_TIMING)
        sync_event.record(subslvr.stream)

        # Retain references to CPU pinned  and GPU device memory
        # until the above enqueued operations have been performed.
        pool_refs.append(X2_gpu_ary)
        pool_refs.append(sub_X2)

        return (sync_event, sub_X2,
            pool_refs,
            cpu_slice_map.copy(),
            gpu_slice_map.copy())

    def initialise(self):
        """ Initialise the sub-solver """

        def _init_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.initialise()

        if not self.initialised:
            for ex in self.executors:
                ex.submit(_init_func).result()

            self.initialised = True

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        def _sync_wait(future):
            """
            Return a copy of the pinned chi-squared after
            synchronizing on the cuda_event
            """
            cuda_event, pinned_X2, pool_refs, cpu, gpu = future.result()

            for k, s in cpu.iteritems():
                cpu[k] = '[{b}, {e}]'.format(b=s.start, e=s.stop)

            for k, s in gpu.iteritems():
                gpu[k] = '[{b}, {e}]'.format(b=s.start, e=s.stop)

            try:
                cuda_event.synchronize()
            except cuda.LogicError as e:
                import json
                print 'GPU', json.dumps(gpu, indent=2)
                print 'CPU', json.dumps(cpu, indent=2)
                raise e, None, sys.exc_info()[2]

            return pinned_X2.copy()

        # For easier typing
        C = CompositeBiroSolver
        zipped_ex = zip(self.executors, self.sync_executors)

        # Running sum of the chi-squared values returned in futures
        X2_sum = self.ft(0.0)

        # Sets of return value futures for each executor
        value_futures = [set() for ex in self.executors]

        # Iterate over the visibility space, i.e. slices over
        # the CPU and GPU arrays
        for cpu_slice_map, gpu_slice_map in self._gen_vis_slices():
            # Attempt to submit work to an executor
            # Poor man's load balancer
            submitted = False
            values_waiting = 0

            while not submitted:
                for i, (enq_ex, sync_ex) in enumerate(zipped_ex):
                    nvalues = len(value_futures[i])
                    values_waiting += nvalues
                    # Too much work on this queue, try another executor
                    if nvalues > self.throttle_factor:
                        continue

                    # Enqueue CUDA operations for solving
                    # this visibility chunk. Future contains:
                    #
                    # (1) A CUDA event which fires after all CUDA operations
                    #     are complete
                    # (2) a pinned memory array that will hold the
                    #     chi-squared total for the chunk after the
                    #     event has fired
                    enqueue_future = enq_ex.submit(
                        C._thread_enqueue_solve_batch,
                        self, cpu_slice_map, gpu_slice_map)

                    # In a synchronisation thread, wait on the
                    # CUDA event and return the array now holding
                    # the chi-squared value.
                    value_future = sync_ex.submit(_sync_wait,
                        enqueue_future)

                    # Throw our futures on the set
                    value_futures[i].add(value_future)

                    # This section of work has been submitted,
                    # break out of the for loop
                    submitted = True
                    break

                # If submission failed, then there are many values
                # still waiting to be computed. Wait for some
                # to finish before attempting work submission
                if not submitted:
                    # Wait for 2/3's of the values
                    threshold = self.throttle_factor*len(self.executors)*2/3.0
                    future_list = [f for f in itertools.chain(*value_futures)]

                    for i, f in enumerate(cf.as_completed(future_list)):
                        # Remove the future from waiting future sets
                        for s in value_futures:
                            s.remove(f)

                        # Add future result to running X2 squared sum
                        X2_sum += f.result()

                        # Break out if we've removed the prescribed
                        # number of futures
                        if i > threshold:
                            break

        # Wait for any remaining values                            
        done, not_done = cf.wait(
            [f for f in itertools.chain(*value_futures)],
            return_when=cf.ALL_COMPLETED)

        # Sum them
        X2_sum += sum(f.result() for f in done)

        # Set the chi-squared value possibly
        # taking the weight vector into account
        if self.use_weight_vector():
            self.set_X2(X2_sum)
        else:
            self.set_X2(X2_sum)/self.sigma_sqrd

    def shutdown(self):
        """ Shutdown the solver """
        def _shutdown_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.shutdown()

            self._thread_shutdown()

        for ex in self.executors:
            ex.submit(_shutdown_func).result()

        for ex in self.sync_executors:
            ex.submit(lambda: self._thread_shutdown()).result()

        self.initialised = False

    def _thread_property_setter(self, name, value):
        for subslvr in self.thread_local.solvers:
            setter_method_name = self.setter_name(name)
            setter_method = getattr(subslvr, setter_method_name)
            setter_method(value)

    def _get_setter_method(self,name):
        """
        Setter method for CompositeBiroSolver properties. Sets the property
        on sub-solvers.
        """

        def setter(self, value):
            # Set the property on the current
            setattr(self, name, value)

            # Then set the property on solver's associated with each
            # executor
            for ex in self.executors:
                ex.submit(CompositeBiroSolver._thread_property_setter,
                    self, name, value)

        return types.MethodType(setter,self)

    def _get_sub_transfer_method(self,name):
        def f(self, npary):
            raise Exception, 'Its illegal to call set methods on the sub-solvers'
        return types.MethodType(f,self)
