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

    def __transfer_slice(self, r, cpu_ary, cpu_idx, gpu_ary, stream):
        cpu_slice = cpu_ary[cpu_idx].squeeze()
        gpu_ary = gpu_ary.squeeze()

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

        print 'Transferring %s with size %s shapes [%s vs %s]' % (
            r.name, mbu.fmt_bytes(copy_ary.nbytes), copy_ary.shape, gpu_ary.shape)

        gpu_ary.set_async(copy_ary, stream=stream)


    def transfer_arrays(self, sub_solver_idx, time_slice,
        bl_slice, ant0_slice, ant1_slice, chan_slice, src_slice):
        """
        Transfer CPU arrays on the CompositeBiroSolver over to the
        BIRO sub-solvers asynchronously.
        """
        i = sub_solver_idx
        subslvr = self.solvers[i]
        stream = self.stream[i]
        all_slice = slice(None,None,1)

        # Sanity check the baseline and antenna slice differences
        bl_diff = bl_slice.stop - bl_slice.start
        ant0_diff = ant0_slice.stop - ant0_slice.start
        ant1_diff = ant1_slice.stop - ant1_slice.start

        if bl_diff == self.nbl:
            assert ant0_diff == self.na and ant1_diff == self.na, \
                ('Antenna slice lengths must be equal to the number of antenna, '
                'given that the baseline slice length is the number of baselines.')
        elif bl_diff == 1:
            assert ant0_diff == 1 and ant1_diff == 1, \
                ('Antenna slice lengths must be equal one, '
                'given that the baseline slice length is one.')
        else:
            raise ValueError('Baseline slice must be either equal '
                'to the number of baselines, or 1.')

        cpu_slice_map = {
            'ntime': time_slice,
            'nbl': bl_slice,
            'na': ant0_slice,
            'nchan': chan_slice,
            'nsrc': src_slice
        }

        with subslvr.context:
            for r in self.arrays.itervalues():
                # Is there anything to transfer for this array?
                if not r.cpu:
                    print '%s has no CPU array' % r.name
                    continue

                cpu_name = mbu.cpu_name(r.name)
                gpu_name = mbu.gpu_name(r.name)

                # Get the CPU array on the composite solver
                # and the CPU array and the GPU array
                # on the sub-solver
                cpu_ary = getattr(self,cpu_name)
                gpu_ary = getattr(subslvr,gpu_name)

                if gpu_ary is None:
                    print 'Skipping %s' % r.name
                    continue

                # Checking if we're handling two antenna here
                # A precursor to the vile hackery that follows
                try:
                    na_idx = r.sshape.index('na')
                    two_ant_case = (subslvr.na == 2)
                except ValueError:
                    na_idx = 0
                    two_ant_case = False

                # Force using the first antenna slice on each iteration
                cpu_slice_map['na'] = ant0_slice

                # If we've got the two antenna case, slice
                # the gpu array to point to the first antenna position
                if two_ant_case is True:
                    gpu_idx = [0 if i <= na_idx else all_slice
                        for i in range(len(r.shape))]
                    gpu_ary = gpu_ary[tuple(gpu_idx)]
                    assert gpu_ary.flags.c_contiguous is True

                # Set up the slicing of the main CPU array. Map dimensions in cpu_slice_map
                # to the slice arguments, otherwise, just take everything in the dimension
                cpu_idx = tuple([cpu_slice_map[s] if s in cpu_slice_map else all_slice
                    for s in r.sshape])

                self.__transfer_slice(r, cpu_ary, cpu_idx, gpu_ary, stream)

                # Right, handle transfer of the second antenna's data
                if two_ant_case is True:
                    gpu_idx[na_idx] = 1
                    gpu_ary = getattr(subslvr, gpu_name)[tuple(gpu_idx)]
                    assert gpu_ary.flags.c_contiguous is True

                    cpu_slice_map['na'] = ant1_slice

                    # Set up the slicing of the main CPU array.
                    # Map dimensions in cpu_slice_map to the slice arguments,
                    # otherwise, just take everything in the dimension
                    cpu_idx = tuple([cpu_slice_map[s] if s in cpu_slice_map
                        else all_slice for s in r.sshape])

                    self.__transfer_slice(r, cpu_ary, cpu_idx, gpu_ary, stream)

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

    def __gen_rime_sections(self):
        for t in xrange(0, self.ntime, self.time_diff):
            time_slice = slice(t, t + self.time_diff, 1)
            for bl in xrange(0, self.nbl, self.bl_diff):
                bl_slice = slice(bl, bl + self.bl_diff, 1)
                for ch in xrange(0, self.nchan, self.chan_diff):
                    chan_slice = slice(ch, ch + self.chan_diff, 1)
                    for src in xrange(0, self.nsrc, self.src_diff):
                        src_slice = slice(src, src + self.src_diff, 1)
                        yield (time_slice, bl_slice, chan_slice, src_slice)

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

    def solve(self):
        """ Solve the RIME """
        if not self.initialised:
            self.initialise()

        nr_var_counts = mbu.sources_to_nr_vars(self.slvr_cfg[Options.SOURCES])
        subslvr_gen = self.__gen_sub_solvers()

        for t, bl, ch, src in self.__gen_rime_sections():
            i, subslvr = subslvr_gen.next()
            src_counts = mbu.source_range(src.start, src.stop, nr_var_counts)
            print 't %s bl %s ch %s src %s src_counts %s' % (t,bl,ch,src,src_counts)
            with subslvr.context:
                if self.bl_diff == self.nbl:
                    ant0 = slice(0, self.na, 1)
                    ant1 = slice(0, self.na, 1)
                elif self.bl_diff == 1:
                    assert subslvr.na == 2, (
                        'One baseline, but number of antenna '
                        'is not equal to 2 (%d)') % (subslvr.na)
                    a0 = self.ant_pairs_cpu[0, t.start, bl.start]
                    ant0 = slice(a0, a0+1, 1)
                    a1 = self.ant_pairs_cpu[1, t.start, bl.start]
                    ant1 = slice(a1, a1+1, 1)

                self.transfer_arrays(i, t, bl, ant0, ant1, ch, src)
                subslvr.rime_e_beam.execute(subslvr, self.stream[i])
                subslvr.rime_b_sqrt.execute(subslvr, self.stream[i])
                subslvr.rime_ekb_sqrt.execute(subslvr, self.stream[i])
                subslvr.rime_sum.execute(subslvr, self.stream[i])

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
