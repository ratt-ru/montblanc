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
from transitions import Machine
import types

import montblanc
import montblanc.util as mbu

from montblanc.BaseSolver import BaseSolver
from montblanc.BaseSolver import DEFAULT_NA
from montblanc.BaseSolver import DEFAULT_NCHAN
from montblanc.BaseSolver import DEFAULT_NTIME
from montblanc.BaseSolver import DEFAULT_NPSRC
from montblanc.BaseSolver import DEFAULT_NGSRC
from montblanc.BaseSolver import DEFAULT_NSSRC
from montblanc.BaseSolver import DEFAULT_DTYPE

import montblanc.impl.biro.v2.BiroSolver as BSV2mod

from montblanc.impl.biro.v3.BiroSolver import BiroSolver

from montblanc.impl.biro.v3.gpu.RimeEK import RimeEK
from montblanc.impl.biro.v3.gpu.RimeGaussBSum import RimeGaussBSum
from montblanc.pipeline import Pipeline

def get_pipeline(**kwargs):
    wv = kwargs.get('weight_vector', False)
    return Pipeline([RimeEK(), RimeGaussBSum(weight_vector=wv)])

ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

class CompositeBiroSolver(BaseSolver):
    """
    Composite solver implementation for BIRO.

    Implements a solver composed of multiple BiroSolvers. The sub-solver
    memory transfers and pipelines are executed asynchronously.
    """
    def __init__(self, na=DEFAULT_NA, nchan=DEFAULT_NCHAN, ntime=DEFAULT_NTIME,
        npsrc=DEFAULT_NPSRC, ngsrc=DEFAULT_NGSRC, nssrc=DEFAULT_NSSRC, dtype=DEFAULT_DTYPE,
        pipeline=None, **kwargs):
        """
        CompositeBiroSolver Constructor

        Parameters:
            na : integer
                Number of antennae.
            nchan : integer
                Number of channels.
            ntime : integer
                Number of timesteps.
            npsrc : integer
                Number of point sources.
            ngsrc : integer
                Number of gaussian sources.
            nssrc : integer
                Number of sersic sources.
            dtype : np.float32 or np.float64
                Specify single or double precision arithmetic.
            pipeline : list of nodes
                nodes defining the GPU kernels used to solve this RIME
        Keyword Arguments:
            context : pycuda.driver.Context
                CUDA context to operate on.
            store_cpu: boolean
                if True, store cpu versions of the kernel arrays
                within the GPUSolver object.
            weight_vector: boolean
                if True, use a weight vector when evaluating the
                RIME, else use a single sigma squared value.
            mem_budget: integer
                Amount of memory in bytes that the solver should
                take into account when fitting the problem onto the
                GPU.
            nsolvers: integer
                Number of sub-solvers to use when subdividing the
                problem.
        """

        pipeline = BSV2mod.get_pipeline(**kwargs) if pipeline is None else pipeline

        super(CompositeBiroSolver, self).__init__(na=na, nchan=nchan, ntime=ntime,
            npsrc=npsrc, ngsrc=ngsrc, nssrc=nssrc, dtype=dtype, **kwargs)

        A_main = copy.deepcopy(BSV2mod.A)
        P_main = copy.deepcopy(BSV2mod.P)

        # Add a custom transfer method for transferring
        # arrays to the sub-solver. Also, in general,
        # only maintain CPU arrays on the main solver,
        # but not GPU arrays, which will exist on the sub-solvers
        for name, ary in A_main.iteritems():
            ary['transfer_method'] = self.get_transfer_method(name)
            ary['gpu'] = False
            ary['cpu'] = True

        for name, prop in P_main.iteritems():
            prop['setter_method'] = self.get_setter_method(name)

        # Do not main CPU versions of result arrays
        A_main['jones_scalar']['cpu'] = False
        A_main['vis']['cpu'] = False
        A_main['chi_sqrd_result']['cpu'] = False

        self.register_arrays(A_main)
        self.register_properties(P_main)

        #print 'Composite Solver Memory CPU %s GPU %s ntime %s' \
        #    % (mbu.fmt_bytes(self.cpu_bytes_required()),
        #    mbu.fmt_bytes(self.gpu_bytes_required()),
        #    ntime)

        # Allocate CUDA constructs using the supplied context
        with self.context as ctx:
            (free_mem,total_mem) = cuda.mem_get_info()

            #(free_mem,total_mem) = cuda.mem_get_info()
            #print 'free %s total %s ntime %s' \
            #    % (mbu.fmt_bytes(free_mem),
            #        mbu.fmt_bytes(total_mem),
            #        ntime)

            # Work with a supplied memory budget, otherwise use
            # free memory less a small amount
            mem_budget = kwargs.get('mem_budget', free_mem-100*ONE_MB)

            # Work out how many timesteps we can fit in our memory budget
            self.vtime = mbu.viable_timesteps(mem_budget,
                self.arrays, self.get_properties())

            (free_mem,total_mem) = cuda.mem_get_info()
            #print 'free %s total %s ntime %s vtime %s' \
            #    % (mbu.fmt_bytes(free_mem),
            #        mbu.fmt_bytes(total_mem),
            #        ntime, self.vtime)

            # They may fit in completely
            if ntime < self.vtime: self.vtime = ntime

            # Configure the number of solvers used
            self.nsolvers = kwargs.get('nsolvers', 4)
            self.time_begin = np.arange(self.nsolvers)*self.vtime//self.nsolvers
            self.time_end = np.arange(1,self.nsolvers+1)*self.vtime//self.nsolvers
            self.time_diff = self.time_end - self.time_begin

            #print 'time begin %s' % self.time_begin
            #print 'time end %s' % self.time_end
            #print 'time diff %s' % self.time_end

            # Create streams and events for the solvers
            self.stream = [cuda.Stream() for i in range(self.nsolvers)]
            self.transfer_start = [cuda.Event(cuda.event_flags.DISABLE_TIMING)
                for i in range(self.nsolvers)]
            self.transfer_end = [cuda.Event(cuda.event_flags.DISABLE_TIMING)
                for i in range(self.nsolvers)]

            # Create a memory pool for PyCUDA gpuarray.sum reduction results
            self.dev_mem_pool = pycuda.tools.DeviceMemoryPool()
            self.dev_mem_pool.allocate(10*ONE_KB).free()

            # Create a pinned memory pool for asynchronous transfer to GPU arrays
            self.pinned_mem_pool = pycuda.tools.PageLockedMemoryPool()
            self.pinned_mem_pool.allocate(shape=(10*ONE_KB,),dtype=np.int8).base.free()

        # Create the sub-solvers
        self.solvers = [BiroSolver(na=na,
            nchan=nchan, ntime=self.time_diff[i],
            npsrc=npsrc, ngsrc=ngsrc, nssrc=nssrc,
            dtype=dtype,
            **kwargs) for i in range(self.nsolvers)]

        # Register arrays on the sub-solvers
        A_sub = copy.deepcopy(BSV2mod.A)
        P_sub = copy.deepcopy(BSV2mod.P)

        for name, ary in A_sub.iteritems():
            # Add a transfer method
            ary['transfer_method'] = self.get_sub_transfer_method(name)
            ary['cpu'] = False
            ary['gpu'] = True

        for i, slvr in enumerate(self.solvers):
            slvr.register_arrays(A_sub)
            slvr.register_properties(P_sub)
            # Indicate that all numpy arrays on the CompositeSolver
            # have been transferred to the sub-solvers
            slvr.was_transferred = {}.fromkeys(
                [v.name for v in self.arrays.itervalues()], True)

        self.use_weight_vector = kwargs.get('weight_vector', False)
        self.initialised = False

        self.slvr_fsm = SolverFSM(self)
        self.fsm = Machine(self.slvr_fsm, states=states, transitions=transitions,
            initial=TRANSFER_DATA)

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

        self.transfer_start[i].record(stream=stream)

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

        self.transfer_end[i].record(stream=stream)

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
            with self.context as ctx:
                for i, slvr in enumerate(self.solvers):
                    slvr.initialise()

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

        # Execute the finite state machine
        with self.context:
            while not self.slvr_fsm.is_done():
                self.slvr_fsm.next()

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
        BSV2mod.BiroSolver.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BSV2mod.BiroSolver.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BSV2mod.BiroSolver.__dict__['get_ap_idx']

# Finite State Machine Implementation

TRANSFER_X2 = 'transfer_x2'
SHOULD_TRANSFER_X2 = 'should_transfer_x2'
TRANSFER_DATA = 'transfer_data'
EK_KERNEL = 'ek'
BSUM_KERNEL = 'bsum'
REDUCE_KERNEL = 'reduce'
NEXT_SOLVER = 'next_solver'
ENTER_FINAL_LOOP = 'enter_final_loop'
FINAL_NEXT_SOLVER = 'final_next_solver'
FINAL_TRANSFER_X2 = 'final_transfer_x2'
DONE = 'done'

states = [SHOULD_TRANSFER_X2, TRANSFER_X2, TRANSFER_DATA,
    EK_KERNEL, BSUM_KERNEL, REDUCE_KERNEL, NEXT_SOLVER,
    ENTER_FINAL_LOOP, FINAL_NEXT_SOLVER, FINAL_TRANSFER_X2,
    DONE]

transitions = [
    {
        'trigger': 'next',
        'source': TRANSFER_DATA,
        'dest': EK_KERNEL,
        'before': 'do_transfer_arrays'
    },
    {
        'trigger': 'next',
        'source': EK_KERNEL,
        'dest': BSUM_KERNEL,
        'before': 'do_ek_kernel'
    },
    {
        'trigger': 'next',
        'source': BSUM_KERNEL,
        'dest': REDUCE_KERNEL,
        'before': 'do_b_sum_kernel'
    },
    # Once the reduction is complete, we may
    # transition to the next round of transfers
    # and kernel executions, OR,
    # if this is the last round of transfer and executions
    # enter the final loop
    {
        'trigger': 'next',
        'source': REDUCE_KERNEL,
        'dest': NEXT_SOLVER,
        'before': 'do_reduction_kernel',
        'unless': 'at_end'
    },
    {
        'trigger': 'next',
        'source': REDUCE_KERNEL,
        'dest': ENTER_FINAL_LOOP,
        'before': 'do_reduction_kernel',
        'conditions': 'at_end'
    },

    # Advance to the next solver and
    # decide whether we should transfer
    # X2 values
    {
        'trigger': 'next',
        'source': NEXT_SOLVER,
        'dest': SHOULD_TRANSFER_X2,
        'before': 'next_solver'
    },
    # From SHOULD_TRANSFER_X2, we can transition to
    # either TRANSFER_X2 or TRANSFER_DATA.
    # First case occurs if we're not on the first iteration.
    {
        'trigger': 'next',
        'source': SHOULD_TRANSFER_X2,
        'dest': TRANSFER_X2,
        'unless' : 'is_first_iteration'
    },
    # Second case occurs on the first iteration.
    {
        'trigger': 'next',
        'source': SHOULD_TRANSFER_X2,
        'dest': TRANSFER_DATA,
        'conditions' : 'is_first_iteration'
    },

    # In the general case, we always go from
    # transferring the X2 to transferring data
    {
        'trigger': 'next',
        'source': TRANSFER_X2,
        'dest': TRANSFER_DATA,
        'before': 'do_transfer_X2'
    },

    # Here, we have states for handling the
    # final loop, which involves pulling
    # the final X2 values off the GPU.
    # We prepare for this by moving
    # back to the first solver (and timestep)
    # for this iteration
    {
        'trigger': 'next',
        'source': ENTER_FINAL_LOOP,
        'dest': FINAL_TRANSFER_X2,
        'before': 'prepare_final_loop'
    },

    {
        'trigger': 'next',
        'source': FINAL_NEXT_SOLVER,
        'dest': FINAL_TRANSFER_X2,
        'before': 'next_solver',
    },

    {
        'trigger': 'next',
        'source': FINAL_TRANSFER_X2,
        'dest': FINAL_NEXT_SOLVER,
        'before': 'do_transfer_X2',
        'unless': 'at_end'
    },
    {
        'trigger': 'next',
        'source': FINAL_TRANSFER_X2,
        'before': 'do_transfer_X2',
        'dest': DONE,
        'conditions': 'at_end'
    },
]

class SolverFSM:
    def __init__(self, composite_solver):
        self.comp_slvr = slvr = composite_solver
        self.current_slvr = 0
        self.sub_time_diff = np.array([s.ntime for s in slvr.solvers])
        self.sub_time_end = np.cumsum(self.sub_time_diff)
        self.sub_time_begin = self.sub_time_end - self.sub_time_diff
        self.current_time = self.sub_time_begin[self.current_slvr] = 0
        self.current_time_diff = self.get_time_diff(
            self.current_time, self.current_slvr)
        self.iteration = 0

        self.X2_gpu_arys = [None for s in slvr.solvers]

        slvr.X2 = 0.0

        with slvr.context:
            for subslvr in slvr.solvers:
                subslvr.X2_tmp = slvr.pinned_mem_pool.allocate(
                    shape=slvr.X2_shape, dtype=slvr.X2_dtype)

    def get_time_diff(self, current_time, current_slvr):
        # Find the time step difference for the supplied solver.
        # It may be smaller than the number of timesteps supported
        # by the solver, since the actual problem may not fit
        # exactly into the timesteps supported by each solver
        diff = self.sub_time_diff[current_slvr]
        spill = self.comp_slvr.ntime - (current_time + diff)

        if spill < 0:
            diff += spill

        return diff

    def do_transfer_arrays(self):
        # Execute array transfer operations for
        # the EK and B sum kernels on the current stream
        self.comp_slvr.transfer_arrays(self.current_slvr,
            self.current_time, self.current_time + self.current_time_diff)

    def do_ek_kernel(self):
        # Execute the EK kernel on the current stream
        slvr, i = self.comp_slvr, self.current_slvr
        slvr.solvers[i].rime_ek.execute(
            slvr.solvers[i],
            slvr.stream[i])

    def do_b_sum_kernel(self):
        # Execute the B Sum kernel on the current stream
        slvr, i = self.comp_slvr, self.current_slvr
        slvr.solvers[i].rime_b_sum.execute(
            slvr.solvers[i],
            slvr.stream[i])

    def do_reduction_kernel(self):
        # Execute the reduction kernel on the current stream
        slvr, i = self.comp_slvr, self.current_slvr
        subslvr, t_diff = self.comp_slvr.solvers[i], self.current_time_diff
        # Swap the allocators for the reduction, since it uses
        # chi_sqrd_result_gpu's allocator internally
        tmp_alloc = subslvr.chi_sqrd_result_gpu.allocator
        subslvr.chi_sqrd_result_gpu.allocator = slvr.dev_mem_pool.allocate
        # OK, perform the reduction over the appropriate timestep section
        # of the chi squared terms.
        self.X2_gpu_arys[i] = pycuda.gpuarray.sum(
            subslvr.chi_sqrd_result_gpu[:t_diff,:],
            stream=slvr.stream[i])
        # Repair the allocator
        subslvr.chi_sqrd_result_gpu.allocator = tmp_alloc

    def do_transfer_X2(self):
        # Execute the transfer of the Chi-Squared value to the CPU
        # on the current stream. Also synchronises on this stream
        # and adds this value to the Chi-Squared total.
        slvr, i = self.comp_slvr, self.current_slvr
        subslvr = self.comp_slvr.solvers[i]
        self.X2_gpu_arys[i].get_async(
            ary=slvr.solvers[i].X2_tmp,
            stream=slvr.stream[i])

        slvr.stream[i].synchronize()

        # Divide by sigma squared if necessary
        if not slvr.use_weight_vector:
            slvr.X2 += subslvr.X2_tmp[0] / subslvr.sigma_sqrd
        else:
            slvr.X2 += subslvr.X2_tmp[0]

    def next_solver(self):
        # Move to the next solver
        self.current_time += self.current_time_diff
        self.current_slvr = (self.current_slvr + 1) % self.comp_slvr.nsolvers
        self.current_time_diff = self.get_time_diff(
            self.current_time, self.current_slvr)

        if self.current_slvr == 0:
            self.iteration += 1

    def prev_solver(self):
        # Move to the previous solver
        prev_slvr = (self.current_slvr - 1) % self.comp_slvr.nsolvers
        self.current_time -= self.sub_time_diff[prev_slvr]
        self.current_slvr = prev_slvr
        self.current_time_diff = self.sub_time_diff[prev_slvr]

        if self.current_slvr == self.comp_slvr.nsolvers - 1:
            self.iteration -= 1

    def prepare_final_loop(self):
        # Prepare for the final loop by
        # rewinding until we reach the first solver
        while self.current_slvr > 0:
            self.prev_solver()

    def is_first_iteration(self):
        return self.iteration == 0

    def is_last_iteration(self):
        time_end = self.current_time + self.sub_time_diff[self.current_time:].sum()
        return time_end >= self.comp_slvr.ntime

    def is_last_solver(self):
        return self.current_slvr == self.comp_slvr.nsolvers - 1

    def at_end(self):
        return self.current_time + self.current_time_diff >= self.comp_slvr.ntime

    def __str__(self):
        return 'iteration %d solver %d time %d diff %d ' % \
            (self.iteration, self.current_slvr, self.current_time,
            self.current_time_diff, )
