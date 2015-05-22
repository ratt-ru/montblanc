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
import pycuda.gpuarray
from transitions import Machine

# Finite State Machine Implementation
START = 'start'
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

states = [START, SHOULD_TRANSFER_X2, TRANSFER_X2, TRANSFER_DATA,
    EK_KERNEL, BSUM_KERNEL, REDUCE_KERNEL, NEXT_SOLVER,
    ENTER_FINAL_LOOP, FINAL_NEXT_SOLVER, FINAL_TRANSFER_X2,
    DONE]

transitions = [
    {
        'trigger': 'next',
        'source': START,
        'dest': TRANSFER_DATA,
        'before': 'do_start',
    },

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
        'after': 'do_done',
        'dest': DONE,
        'conditions': 'at_end'
    },
]

class FsmSolver(Machine):
    def __init__(self, composite_solver):
        Machine.__init__(self, states=states, 
            transitions=transitions, initial=START)
        self.comp_slvr = slvr = composite_solver
        self.sub_time_diff = np.array([s.ntime for s in slvr.solvers])
        self.sub_time_end = np.cumsum(self.sub_time_diff)
        self.sub_time_begin = self.sub_time_end - self.sub_time_diff

        self.X2_gpu_arys = [None for s in slvr.solvers]

        with slvr.context:
            for subslvr in slvr.solvers:
                subslvr.X2_tmp = slvr.pinned_mem_pool.allocate(
                    shape=slvr.X2_shape, dtype=slvr.X2_dtype)

    def do_start(self):
        self.current_slvr = 0
        self.current_time = 0
        self.current_time_diff = self.get_time_diff(
            self.current_time, self.current_slvr)
        self.iteration = 0
        self.comp_slvr.X2 = 0.0

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
            subslvr.chi_sqrd_result_gpu[:t_diff,:,:],
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
        # rewinding N-1 solvers
        i, N = 1, self.comp_slvr.nsolvers

        while i < N and self.current_time > 0:
            self.prev_solver()
            i += 1

    def do_done(self):
		pass

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
