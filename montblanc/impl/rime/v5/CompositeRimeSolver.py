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

from collections import defaultdict
from copy import (copy as shallowcopy, deepcopy)
import functools
import itertools
import numpy as np
import types
import sys

import concurrent.futures as cf
import threading

import pycuda.driver as cuda
import pycuda.tools

import hypercube.util as hcu

import montblanc
import montblanc.util as mbu

from montblanc.solvers import MontblancNumpySolver
from montblanc.config import RimeSolverConfig as Options

from montblanc.impl.rime.v4.config import (
    A as v4Arrays,
    P as v4Props,
    Classifier)

import montblanc.impl.rime.v4.RimeSolver as BSV4mod

from montblanc.impl.rime.v5.RimeSolver import RimeSolver

ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

ASYNC_HTOD = 'htod'
ASYNC_DTOH = 'dtoh'

ALL_SLICE = slice(None,None,1)
EMPTY_SLICE = slice(0,0,1)

ORDERING_CONSTRAINTS = { nr_var : 1 for nr_var in mbu.source_nr_vars() }
ORDERING_CONSTRAINTS.update({ 'nsrc' : 1,
    'ntime': 2, 'nbl': 3, 'na': 3, 'nchan': 4 })

ORDERING_RANK = [' or '.join(['nsrc'] + mbu.source_nr_vars()),
    'ntime', ' or '.join(['nbl', 'na']), 'nchan']

def _update_refs(pool, new_refs):
    for key, value in new_refs.iteritems():
        pool[key].extend(value)

class CompositeRimeSolver(MontblancNumpySolver):
    """
    Composite solver implementation for RIME.

    Implements a solver composed of multiple RimeSolvers. The sub-solver
    memory transfers and pipelines are executed asynchronously.
    """
    def __init__(self, slvr_cfg):
        """
        RimeSolver Constructor

        Parameters:
            slvr_cfg : SolverConfiguration
                Solver Configuration variables
        """
        super(CompositeRimeSolver, self).__init__(slvr_cfg=slvr_cfg)

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
        from montblanc.impl.rime.v4.ant_pairs import monkey_patch_antenna_pairs
        monkey_patch_antenna_pairs(self)

        # Copy the v4 arrays and properties and
        # modify them for use on this Composite Solver
        A_main, P_main = self._cfg_comp_slvr_arys_and_props(v4Arrays, v4Props)

        self.register_properties(P_main)
        self.register_arrays(A_main)

        # Look for ignored and supplied arrays in the solver configuration
        array_cfg = slvr_cfg.get('array_cfg', {})
        ignore = array_cfg.get('ignore', None)
        supplied = array_cfg.get('supplied', None)

        # Create arrays on the solver, ignoring
        # and using supplied arrays as necessary
        self.create_arrays(ignore, supplied)

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
        C = CompositeRimeSolver

        # Create a one thread executor for each device context,
        # i.e. a thread per device
        enqueue_executors = [cf.ThreadPoolExecutor(1) for ctx in self.dev_ctxs]
        sync_executors = [cf.ThreadPoolExecutor(1) for ctx in self.dev_ctxs]

        self.enqueue_executors = enqueue_executors
        self.sync_executors = sync_executors
        self.initialised = False
        self._vis_write_mode = slvr_cfg.get(Options.VISIBILITY_WRITE_MODE)

        montblanc.log.info('Created {d} executor(s).'.format(d=len(enqueue_executors)))

        # Initialise executor threads
        for ex, ctx in zip(enqueue_executors, self.dev_ctxs):
            ex.submit(C._thread_init, self, ctx).result()

        for ex, ctx in zip(sync_executors, self.dev_ctxs):
            ex.submit(C._thread_init, self, ctx).result()

        montblanc.log.info('Initialised {d} thread(s).'.format(d=len(enqueue_executors)))

        # Get a template dictionary
        T = self.template_dict()

        A_sub, P_sub = self._cfg_sub_slvr_arys_and_props(v4Arrays, v4Props)
        self._validate_arrays(A_sub)

        # Find the budget with the lowest memory usage
        # Work with the device with the lowest memory
        budgets = sorted([ex.submit(C._thread_budget, self,
                            slvr_cfg, A_sub, T).result()
                        for ex in enqueue_executors],
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

        subslvr_cfg = self._cfg_subslvr_dims(subslvr_cfg, P)

        # Extract the dimension differences
        self.src_diff = P[Options.NSRC]
        self.time_diff = P[Options.NTIME]
        self.ant_diff = P[Options.NA]
        self.bl_diff = P[Options.NBL]
        self.chan_diff = P[Options.NCHAN]

        montblanc.log.info('Creating {s} solver(s) on {d} device(s).'
            .format(s=nsolvers, d=len(enqueue_executors)))

        # Now create the solvers on each thread
        for ex in enqueue_executors:
            ex.submit(C._thread_create_solvers,
                self, subslvr_cfg, P, nsolvers).result()

        montblanc.log.info('Solvers Created')

        # Register arrays and properties on each thread's solvers
        for ex in enqueue_executors:
            ex.submit(C._thread_reg_sub_arys_and_props,
                self, A_sub, P_sub).result()

        montblanc.log.info('Priming Memory Pools')

        # Prime the memory pools on each sub-solver
        for ex in enqueue_executors:
            ex.submit(C._thread_prime_memory_pools, self).result()

    def _gen_source_slices(self):
        """
        Iterate over the visibility space in chunks, returning a
        dictionary of slices keyed on the following dimensions:
            nsrc, npsrc, ngsrc, nssrc, ...
        """

        # Get a list of source number variables/dimensions
        src_nr_vars = mbu.source_nr_vars()
        lower_extents = self.dim_lower_extent(*src_nr_vars)
        upper_extents = self.dim_upper_extent(*src_nr_vars)

        non_zero = [(n, l) for n, l
            in zip(src_nr_vars, lower_extents) if l != 0]

        if len(non_zero) > 0:
            raise ValueError("The following source dimensions "
                "have non-zero lower extents [{nzd}]".format(
                    nzd=non_zero))

        # Create source counts, or dimension extent sizes
        # for each source type/dimension
        src_nr_var_counts = { nr_var: u-l
            for nr_var, l, u in zip(src_nr_vars,
                lower_extents, upper_extents) }

        # Work out which range of sources in the total space
        # we are iterating over
        nsrc_lower, nsrc_upper = self.dim_extents(Options.NSRC)

        # Create the slice dictionaries, which we use to index
        # dimensions of the CPU and GPU array.
        cpu_slice, gpu_slice = {}, {}

        # Set up source slicing
        for src in xrange(nsrc_lower, nsrc_upper, self.src_diff):
            src_end = min(src + self.src_diff, nsrc_upper)
            src_diff = src_end - src
            cpu_slice[Options.NSRC] = slice(src, src_end, 1)
            gpu_slice[Options.NSRC] = slice(0, src_diff, 1)

            # Get the source slice ranges for each individual
            # source type, and update the CPU dictionary with them
            src_range_slices = mbu.source_range_slices(
                src, src_end, src_nr_var_counts)
            cpu_slice.update(src_range_slices)

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

        ((ntime_lower, ntime_upper), (nbl_lower, nbl_upper),
            (na_lower, na_upper),
            (nchan_lower, nchan_upper)) = self.dim_extents(
                'ntime', 'nbl', 'na', 'nchan')

        npol = self.dim_global_size('npol')

        # Create the slice dictionaries, which we use to index
        # dimensions of the CPU and GPU array.
        cpu_slice, gpu_slice = {}, {}

        # Visibilities are a dependent, convenience dimension
        # Set the slice to a noop
        cpu_slice['nvis'] = slice(0, 0, 1)
        gpu_slice['nvis'] = slice(0, 0, 1)

        montblanc.log.debug('Generating RIME slices')

        # Set up time slicing
        for t in xrange(ntime_lower, ntime_upper, self.time_diff):
            t_end = min(t + self.time_diff, ntime_upper)
            t_diff = t_end - t
            cpu_slice[Options.NTIME] = slice(t,  t_end, 1)
            gpu_slice[Options.NTIME] = slice(0, t_diff, 1)

            # Set up baseline and antenna slicing
            for bl in xrange(nbl_lower, nbl_upper, self.bl_diff):
                bl_end = min(bl + self.bl_diff, nbl_upper)
                bl_diff = bl_end - bl
                cpu_slice[Options.NBL] = slice(bl,  bl_end, 1)
                gpu_slice[Options.NBL] = slice(0, bl_diff, 1)

                # Take all antenna pairs
                cpu_slice[Options.NA] = slice(na_lower, na_upper, 1)
                gpu_slice[Options.NA] = slice(na_lower, na_upper, 1)

                # Set up channel slicing
                for ch in xrange(nchan_lower, nchan_upper, self.chan_diff):
                    ch_end = min(ch + self.chan_diff, nchan_upper)
                    ch_diff = ch_end - ch
                    cpu_slice[Options.NCHAN] = slice(ch, ch_end, 1)
                    gpu_slice[Options.NCHAN] = slice(0, ch_diff, 1)

                    # Polarised Channels are a dependent, convenience dimension
                    # equal to number of channels x number of polarisations
                    cpu_slice['npolchan'] = slice(ch*npol, ch_end*npol, 1)
                    gpu_slice['npolchan'] = slice(0, ch_diff*npol, 1)

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
            name = dim.name
            if name in P:
                # Copy dimension data for reconfiguration
                sub_dim = shallowcopy(dim)
                sub_dim.update(local_size=P[name],
                    lower_extent=0, upper_extent=P[name])

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

    def _enqueue_array_slice(self, r, subslvr,
        cpu_ary, cpu_idx, gpu_ary, gpu_idx, direction, dirty):
        """
        Copies a slice of the CPU array into a slice of the GPU array.

        Returns pinned memory array or None if no transfer
        took place.
        """

        # Check if this slice exists in dirty.
        # If so, it has already been transferred and
        # we can ignore it
        cache_idx = dirty.get(r.name, None)

        if cache_idx is not None and cache_idx == cpu_idx:
            #montblanc.log.debug("Cache hit on {n} index {i} "
            #        .format(n=r.name, i=cpu_idx))

            return None

        # No dirty entry, or we're going to
        # replace the existing entry.
        dirty[r.name] = cpu_idx

        cpu_slice = cpu_ary[cpu_idx].squeeze()
        gpu_ary = gpu_ary[gpu_idx].squeeze()

        # Obtain some pinned memory from the memory pool
        pinned_ary = subslvr.pinned_mem_pool.allocate(
            shape=gpu_ary.shape, dtype=gpu_ary.dtype)

        if direction == ASYNC_HTOD:
            # Copy data into pinned memory and enqueue the transfer.
            if pinned_ary.ndim > 0:
                pinned_ary[:] = cpu_slice
            else:
                pinned_ary[()] = cpu_slice

            gpu_ary.set_async(pinned_ary, stream=subslvr.stream)
        elif direction == ASYNC_DTOH:
            # Enqueue transfer from device into pinned memory.
            gpu_ary.get_async(ary=pinned_ary, stream=subslvr.stream)
        else:
            raise ValueError("Invalid direction '{d}'".format(d=direction))

        return pinned_ary

    def _enqueue_array(self, subslvr,
        cpu_slice_map, gpu_slice_map,
        direction=None, dirty=None, classifiers=None):
        """
        Enqueue asynchronous copies from CPU arrays on the
        CompositeRimeSolver to GPU arrays on the RIME sub-solvers
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

        Returns a list of arrays allocated with pinned memory.
        Entries in this list may be None.
        """
        pool_refs = defaultdict(list)

        if direction is None:
            direction = ASYNC_HTOD

        if dirty is None:
            dirty = {}
        
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
                for s in r.shape]

            gpu_idx = [gpu_slice_map[s]
                if s in gpu_slice_map else ALL_SLICE
                for s in r.shape]

            # Bail if there's an empty slice in the index
            if gpu_idx.count(EMPTY_SLICE) > 0:
                #print '%s has an empty slice, skipping' % r.name
                continue

            pinned_ary = self._enqueue_array_slice(r, subslvr,
                cpu_ary, cpu_idx, gpu_ary, tuple(gpu_idx),
                direction, dirty)

            if pinned_ary is not None:
                pool_refs[r.name].append(pinned_ary)

        return pool_refs

    def __enter__(self):
        """
        When entering a run-time context related to this solver,
        initialise and return it.
        """
        self.initialise()
        return self

    def __exit__(self, etype, evalue, etrace):
        """
        When exiting a run-time context related to this solver,
        also perform exit for the sub-solvers.
        """
        try:
            self.shutdown()
        except Exception as se:
            # If there was no exception entering
            # this context, re-raise the shutdown exception
            if evalue is None:
                raise
            else:
                # Otherwise log the shutdown exception
                montblanc.log.exception("Exception occurred during "
                    "ComposeRimeSolver shutdown")

                # And raise the exception from the context
                raise etype, evalue, etrace

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

        if self.thread_local.context is not None:
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

        ntime_split = np.int32(np.ceil(100.0 / nsolvers))
        ntime_split_str = 'ntime={n}'.format(n=ntime_split)

        # Figure out a viable dimension configuration
        # given the total problem size 
        viable, modded_dims = mbu.viable_dim_config(
            mem_budget, A_sub, props,
                [ntime_split_str, src_reduction_str, 'ntime',
                'nbl={na}&na={na}'.format(na=na),
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

        # GPU Device memory pool, used in cases where PyCUDA
        # needs GPU memory that we haven't been able to pre-allocate
        dev_mem_pool = pycuda.tools.DeviceMemoryPool()

        # CPU Pinned memory pool, used for array transfers
        pinned_mem_pool = pycuda.tools.PageLockedMemoryPool()

        # Mutex for guarding the memory pools
        pool_lock = threading.Lock()

        # Dirty index, indicating the CPU index of the
        # data currently on the GPU, used for avoiding
        # array transfer
        self.thread_local.dirty = [{} for n in range(nsolvers)]

        # Configure thread local storage
        # Number of solvers in this thread
        self.thread_local.nsolvers = nsolvers
        # List of solvers used by this thread, set below
        self.thread_local.solvers = [None for s in range(nsolvers)]
        # Initialise the subsolver generator
        self.thread_local.subslvr_gen = self._thread_gen_sub_solvers()

        # Set the CUDA context in the configuration to
        # the one associated with this thread
        subslvr_cfg[Options.CONTEXT] = self.thread_local.context

        # Create solvers for this context
        for i in range(nsolvers):
            subslvr = RimeSolver(subslvr_cfg)

            # Configure the source dimensions of each sub-solver.
            # Change the local size of each source dim so that there is
            # enough space in the associated arrays for NSRC sources.
            # Initially, configure the extents to be [0, NSRC], although
            # this will be setup properly in _thread_solve_sub
            nsrc = P[Options.NSRC]

            U = [{
                'name': nr_var,
                'local_size': nsrc if nsrc < P[nr_var] else P[nr_var],
                'lower_extent': 0,
                'upper_extent': nsrc if nsrc < P[nr_var] else P[nr_var],
            } for nr_var in [Options.NSRC] + mbu.source_nr_vars()]

            subslvr.update_dimensions(U)

            # Give sub solvers access to device and pinned memory pools
            subslvr.dev_mem_pool = dev_mem_pool
            subslvr.pinned_mem_pool = pinned_mem_pool
            subslvr.pool_lock = pool_lock
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
            subslvr.create_arrays()

    def _thread_prime_memory_pools(self):
        """
        We use memory pools to avoid allocating both CUDA
        pinned host and device memory. This function fakes
        allocations prior to running the solver so that
        the memory pools are 'primed' with memory allocations
        that can be re-used during actual execution of the solver
        """

        montblanc.log.debug('Priming memory pools in thread %s',
            threading.current_thread())

        nsrc = self.dim_local_size('nsrc')

        # Retain references to pool allocations
        pinned_pool_refs = defaultdict(list)
        device_pool_refs = defaultdict(list)
        pinned_allocated = 0

        # Class of arrays that are to be transferred
        classifiers = [Classifier.E_BEAM_INPUT,
            Classifier.B_SQRT_INPUT,
            Classifier.EKB_SQRT_INPUT,
            Classifier.COHERENCIES_INPUT]

        # Estimate number of kernels for constant data
        nkernels = len(classifiers)

        # Detect already transferred array chunks
        dirty = {}

        # Get the first chunk of the visibility space
        cpu_slice_map, gpu_slice_map = self._gen_vis_slices().next()

        for i, subslvr in enumerate(self.thread_local.solvers):
            # For the maximum number of visibility chunks that can be enqueued
            for T in range(self.throttle_factor):
                # For each source batch within the visibility chunk
                for cpu_src_slice_map, gpu_src_slice_map in self._gen_source_slices():
                    cpu_slice_map.update(cpu_src_slice_map)
                    gpu_slice_map.update(gpu_src_slice_map)

                    # Allocate pinned memory for transfer arrays
                    # retaining references to them
                    refs = self._enqueue_array(subslvr,
                        cpu_slice_map, gpu_slice_map, 
                        direction=ASYNC_HTOD, dirty=dirty,
                        classifiers=classifiers)
                    pinned_allocated += sum([r.nbytes
                        for l in refs.values() for r in l])
                    _update_refs(pinned_pool_refs, refs)

                    # Allocate pinned memory for constant memory transfers
                    cdata = subslvr.const_data().ndary()

                    for k in range(nkernels):
                        cdata_ref = subslvr.pinned_mem_pool.allocate(
                            shape=cdata.shape, dtype=cdata.dtype)
                        pinned_allocated += cdata_ref.nbytes
                        pinned_pool_refs['cdata'].append(cdata_ref)

                    # Allocate device memory for arrays that need to be
                    # allocated from a pool by PyCUDA's reduction kernels
                    dev_ary = subslvr.dev_mem_pool.allocate(self.X2.nbytes)
                    device_pool_refs['X2_gpu'].append(dev_ary)

        device = self.thread_local.context.get_device()

        montblanc.log.info('Primed pinned memory pool '
            'of size {n} for device {d}.'.format(
                d=device.name(), n=mbu.fmt_bytes(pinned_allocated)))

        # Now force return of memory to the pools
        for key, array_list in pinned_pool_refs.iteritems():
            [a.base.free() for a in array_list]

        for array_list in device_pool_refs.itervalues():
            [a.free() for a in array_list]

    def _thread_start_solving(self):
        """
        Contains enqueue thread functionality that needs to
        take place at the start of each solution
        """

        # Clear the dirty dictionary to force each array to be
        # transferred at least once. e.g. the beam cube
        [d.clear() for d in self.thread_local.dirty]

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

        # A dictionary of references to memory pool allocated objects
        # ensuring that said objects remained allocated until
        # after compute has been performed. Returned from
        # this function, this object should be discarded when
        # reading the result of the enqueued operations.
        pool_refs = defaultdict(list)

        # Cache keyed by array names and contained indices
        # This is used to avoid unnecessary CPU to GPU copies
        # by caching the last index of the CPU array
        dirty = tl.dirty[i]

        # Guard pool allocations with a coarse-grained mutex
        with subslvr.pool_lock:
            # Now, iterate over our source chunks, enqueuing
            # memory transfers and CUDA kernels
            for src_cpu_slice_map, src_gpu_slice_map in self._gen_source_slices():
                # Update our maps with source slice information
                cpu_slice_map.update(src_cpu_slice_map)
                gpu_slice_map.update(src_gpu_slice_map)

                # Configure dimension extents and global size on the sub-solver
                for name, slice_ in cpu_slice_map.iteritems():
                    subslvr.update_dimension(name=name,
                        global_size=self.dimension(name).global_size,
                        lower_extent=slice_.start,
                        upper_extent=slice_.stop)

                # Enqueue E Beam
                kernel = subslvr.rime_e_beam
                new_refs = self._enqueue_array(
                    subslvr, cpu_slice_map, gpu_slice_map,
                    direction=ASYNC_HTOD, dirty=dirty,
                    classifiers=[Classifier.E_BEAM_INPUT])
                cdata_ref = self._enqueue_const_data_htod(
                    subslvr, kernel.rime_const_data[0])
                _update_refs(pool_refs, new_refs)
                _update_refs(pool_refs, {'cdata_ebeam' : [cdata_ref]})
                kernel.execute(subslvr, subslvr.stream)

                # Enqueue B Sqrt
                kernel = subslvr.rime_b_sqrt
                new_refs = self._enqueue_array(
                    subslvr, cpu_slice_map, gpu_slice_map,
                    direction=ASYNC_HTOD, dirty=dirty,
                    classifiers=[Classifier.B_SQRT_INPUT])
                cdata_ref = self._enqueue_const_data_htod(
                    subslvr, kernel.rime_const_data[0])
                _update_refs(pool_refs, new_refs)
                _update_refs(pool_refs, {'cdata_bsqrt' : [cdata_ref]})
                kernel.execute(subslvr, subslvr.stream)

                # Enqueue EKB Sqrt
                kernel = subslvr.rime_ekb_sqrt
                new_refs = self._enqueue_array(
                    subslvr, cpu_slice_map, gpu_slice_map,
                    direction=ASYNC_HTOD, dirty=dirty,
                    classifiers=[Classifier.EKB_SQRT_INPUT])
                cdata_ref = self._enqueue_const_data_htod(
                    subslvr, kernel.rime_const_data[0])
                _update_refs(pool_refs, new_refs)
                _update_refs(pool_refs, {'cdata_ekb' : [cdata_ref]})
                kernel.execute(subslvr, subslvr.stream)

                # Enqueue Sum Coherencies
                kernel = subslvr.rime_sum
                new_refs = self._enqueue_array(
                    subslvr, cpu_slice_map, gpu_slice_map,
                    direction=ASYNC_HTOD, dirty=dirty,
                    classifiers=[Classifier.COHERENCIES_INPUT])
                cdata_ref = self._enqueue_const_data_htod(
                    subslvr, kernel.rime_const_data[0])
                _update_refs(pool_refs, new_refs)
                _update_refs(pool_refs, {'cdata_coherencies' : [cdata_ref]})
                kernel.execute(subslvr, subslvr.stream)

            # Enqueue chi-squared term reduction and return the
            # GPU array allocated to it
            X2_gpu_ary = subslvr.rime_reduce.execute(subslvr, subslvr.stream)

            # Get pinned memory to hold the chi-squared result
            sub_X2 = subslvr.pinned_mem_pool.allocate(
                shape=X2_gpu_ary.shape, dtype=X2_gpu_ary.dtype)
        
            # Enqueue chi-squared copy off the GPU onto the CPU
            X2_gpu_ary.get_async(ary=sub_X2, stream=subslvr.stream)

            # Enqueue transfer of simulator output (model visibilities) to the CPU
            sim_output_refs = self._enqueue_array(subslvr,
                cpu_slice_map, gpu_slice_map,
                direction=ASYNC_DTOH, dirty={},
                classifiers=[Classifier.SIMULATOR_OUTPUT])

        # Should only be model visibilities
        assert len(sim_output_refs) == 1, (
            'Expected one array (model visibilities), '
            'received {l} instead.'.format(l=len(new_refs)))

        model_vis = sim_output_refs['model_vis'][0]

        # Create and record an event directly after the chi-squared copy
        # We'll synchronise on this thread in our synchronisation executor
        sync_event = cuda.Event(cuda.event_flags.DISABLE_TIMING |
            cuda.event_flags.BLOCKING_SYNC)
        sync_event.record(subslvr.stream)

        # Retain references to CPU pinned  and GPU device memory
        # until the above enqueued operations have been performed.
        pool_refs['X2_gpu'].append(X2_gpu_ary)
        pool_refs['X2_cpu'].append(sub_X2)
        pool_refs['model_vis_output'].append(model_vis)

        return (sync_event, sub_X2, model_vis,
            pool_refs, subslvr.pool_lock,
            cpu_slice_map.copy(),
            gpu_slice_map.copy())

    def initialise(self):
        """ Initialise the sub-solver """

        def _init_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.initialise()

        if not self.initialised:
            for ex in self.enqueue_executors:
                ex.submit(_init_func).result()

            self.initialised = True

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        model_vis_sshape = self.arrays()['model_vis']['shape']

        def _free_pool_allocs(pool_refs, pool_lock):
            """ Free pool-allocated objects in pool_refs, guarded by pool_lock """
            import pycuda.driver as cuda
            import pycuda.gpuarray as gpuarray

            cuda_types = (cuda.PooledDeviceAllocation, cuda.PooledHostAllocation)

            debug_str_list = ['Pool de-allocations per array name',]
            debug_str_list.extend('({k}, {l})'.format(k=k,l=len(v))
                for k, v in pool_refs.iteritems())

            montblanc.log.debug(' '.join(debug_str_list))

            with pool_lock:
                for k, ref in ((k, r) for k, rl in pool_refs.iteritems()
                                for r in rl):
                    if isinstance(ref, np.ndarray):
                        ref.base.free()
                    elif isinstance(ref, cuda_types):
                        ref.free()
                    elif isinstance(ref, gpuarray.GPUArray):
                        ref.gpudata.free()
                    else:
                        raise TypeError("Don't know how to release pool allocated "
                            "object '{n}'' of type {t}.".format(n=k, t=type(ref)))

        def _sync_wait(future):
            """
            Return a copy of the pinned chi-squared, pinned model visibilities
            and index into the CPU array after synchronizing on the cuda_event
            """
            cuda_event, pinned_X2, pinned_model_vis, \
                pool_refs, pool_lock, cpu, gpu = future.result()

            try:
                cuda_event.synchronize()
            except cuda.LogicError as e:
                # Format the slices nicely
                pretty_cpu = { k: '[{b}, {e}]'.format(b=s.start, e=s.stop) 
                    for k, s in cpu.iteritems() }
                pretty_gpu = { k: '[{b}, {e}]'.format(b=s.start, e=s.stop) 
                    for k, s in gpu.iteritems() }

                import json
                print 'GPU', json.dumps(pretty_gpu, indent=2)
                print 'CPU', json.dumps(pretty_cpu, indent=2)
                raise e, None, sys.exc_info()[2]

            # Work out the CPU view in the model visibilities
            model_vis_idx = [cpu[s] if s in cpu else ALL_SLICE
                for s in model_vis_sshape]

            # Infer proper model visibility shape, this may
            # have been squeezed when enqueueing the slice
            model_vis_shape = tuple(cpu[s].stop - cpu[s].start
                if s in cpu else s for s in model_vis_sshape)

            X2 = pinned_X2.copy()
            model_vis = pinned_model_vis.copy().reshape(model_vis_shape)

            _free_pool_allocs(pool_refs, pool_lock)

            return X2, model_vis, model_vis_idx

        # For easier typing
        C = CompositeRimeSolver

        # Perform any initialisation required by each
        # thread prior to solving
        for f in cf.as_completed([ex.submit(self._thread_start_solving)
                for ex in self.enqueue_executors]):
            f.result()

        # Create an iterator that cycles through each device's
        # executors, also returning the device index (enumerate)
        ex_it = itertools.cycle(enumerate(
            zip(self.enqueue_executors, self.sync_executors)))

        # Running sum of the chi-squared values returned in futures
        X2_sum = self.ft(0.0)

        # Sets of return value futures for each executor
        value_futures = [set() for ex in self.enqueue_executors]

        # Wait for 1/3 of in-flight futures
        threshold = self.throttle_factor*len(self.enqueue_executors)*1/3.0

        # Decide whether to replace or accumulate model visibilities
        def _overwrite(model_slice, model_chunk):
            model_slice[:] = model_chunk

        def _accumulate(model_slice, model_chunk):
            model_slice[:] += model_chunk

        if self._vis_write_mode == Options.VISIBILITY_WRITE_MODE_SUM:
            vis_write = _accumulate
        else:
            vis_write = _overwrite

        # Iterate over the visibility space, i.e. slices over
        # the CPU and GPU arrays
        for cpu_slice_map, gpu_slice_map in self._gen_vis_slices():
            # Attempt to submit work to an executor
            # Poor man's load balancer
            submitted = False
            values_waiting = 0

            while not submitted:
                # Attempt all devices when submitting work
                for device in range(len(self.enqueue_executors)):
                    # Cycle forward to avoid submitting all work to current device
                    i, (enq_ex, sync_ex) = ex_it.next()
                    nvalues = len(value_futures[i])
                    values_waiting += nvalues
                    # Too much work on this queue, try another executor
                    if nvalues > self.throttle_factor:
                        continue

                    # Enqueue CUDA operations for solving
                    # this visibility chunk. 
                    enqueue_future = enq_ex.submit(
                        C._thread_enqueue_solve_batch,
                        self, cpu_slice_map, gpu_slice_map)

                    # In a synchronisation thread, wait for the
                    # enqueue future to complete and return the
                    # chi-squared value and model visibilities for
                    # this chunk of the visibility space
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
                    future_list = [f for f in itertools.chain(*value_futures)]

                    for i, f in enumerate(cf.as_completed(future_list)):
                        # Remove the future from waiting future sets
                        for s in value_futures:
                            s.discard(f)

                        # Get chi-squared and model visibilities
                        X2, pinned_model_vis, model_vis_idx = f.result()
                        # Sum X2
                        X2_sum += X2
                        # Write model visibilities to the numpy array
                        vis_write(self.model_vis[model_vis_idx], pinned_model_vis[:])

                        # Break out if we've removed the prescribed
                        # number of futures
                        if i >= threshold:
                            break

        # Wait for any remaining values                            
        done, not_done = cf.wait(
            [f for f in itertools.chain(*value_futures)],
            return_when=cf.ALL_COMPLETED)

        # Sum remaining chi-squared
        for f in done:
            # Get chi-squared and model visibilities
            X2, pinned_model_vis, model_vis_idx = f.result()
            # Sum X2
            X2_sum += X2
            # Write model visibilities to the numpy array
            vis_write(self.model_vis[model_vis_idx], pinned_model_vis[:])

        # Set the chi-squared value possibly
        # taking the weight vector into account
        if self.use_weight_vector():
            self.set_X2(X2_sum)
        else:
            self.set_X2(X2_sum/self.sigma_sqrd)

    def shutdown(self):
        """ Shutdown the solver """
        def _shutdown_func():
            for i, subslvr in enumerate(self.thread_local.solvers):
                subslvr.shutdown()
                subslvr.dev_mem_pool.stop_holding()
                subslvr.pinned_mem_pool.stop_holding()

            self._thread_shutdown()

        for ex in self.enqueue_executors:
            ex.submit(_shutdown_func).result()

        for ex in self.sync_executors:
            ex.submit(lambda: self._thread_shutdown()).result()

        self.initialised = False

    def _thread_property_setter(self, name, value):
        for subslvr in self.thread_local.solvers:
            setter_method_name = hcu.setter_name(name)
            setter_method = getattr(subslvr, setter_method_name)
            setter_method(value)

    def _get_setter_method(self,name):
        """
        Setter method for CompositeRimeSolver properties. Sets the property
        on sub-solvers.
        """

        def setter(self, value):
            # Set the property on the current
            setattr(self, name, value)

            # Then set the property on solver's associated with each
            # executor
            for ex in self.enqueue_executors:
                ex.submit(CompositeRimeSolver._thread_property_setter,
                    self, name, value)

        return types.MethodType(setter,self)

    def _get_sub_transfer_method(self,name):
        def f(self, npary):
            raise Exception, 'Its illegal to call set methods on the sub-solvers'
        return types.MethodType(f,self)
