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

import montblanc.impl.biro.v2.BiroSolver as BSV2mod
import montblanc.impl.biro.common

from montblanc.impl.biro.v3.BiroSolver import BiroSolver

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfig as Options)


ONE_KB = 1024
ONE_MB = ONE_KB**2
ONE_GB = ONE_KB**3

class CompositeBiroSolver(BaseSolver):
    """
    Composite solver implementation for BIRO.

    Implements a solver composed of multiple BiroSolvers. The sub-solver
    memory transfers and pipelines are executed asynchronously.
    """
    def __init__(self, slvr_cfg):
        """

        CompositeBiroSolver Constructor

        Parameters:
            slvr_cfg : BiroSolverConfiguration
                Solver Configuration variables
        """

        super(CompositeBiroSolver, self).__init__(slvr_cfg)

        P_main = copy.deepcopy(BSV2mod.P)
        A_main = copy.deepcopy(BSV2mod.A)

        # Add a custom setter method for transferring
        # properties to the sub-solver.
        for prop in P_main:
            prop['setter_method'] = self.get_setter_method(prop['name'])

        # Add a custom transfer method for transferring
        # arrays to the sub-solver. Also, in general,
        # only maintain CPU arrays on the main solver,
        # but not GPU arrays, which will exist on the sub-solvers
        for ary in A_main:
            ary['transfer_method'] = self.get_transfer_method(ary['name'])
            ary['gpu'] = False
            ary['cpu'] = True

        # Add custom property setter method
        for prop in P_main:
            prop['setter_method'] = self.get_setter_method(ary['name'])

        # Do not create CPU versions of result arrays

        for ary in [a for a in A_main if a['name'] in
                ['jones_scalar', 'vis', 'chi_sqrd_result']]:
            ary['cpu'] = False

        self.register_properties(P_main)
        self.register_arrays(A_main)

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
            mem_budget = slvr_cfg.get('mem_budget', free_mem-100*ONE_MB)

            # Work out how many timesteps we can fit in our memory budget
            self.vtime = mbu.viable_timesteps(mem_budget,
                self._arrays, self.template_dict())

            (free_mem,total_mem) = cuda.mem_get_info()
            #print 'free %s total %s ntime %s vtime %s' \
            #    % (mbu.fmt_bytes(free_mem),
            #        mbu.fmt_bytes(total_mem),
            #        ntime, self.vtime)

            # They may fit in completely
            if self.ntime < self.vtime: self.vtime = self.ntime

            # Configure the number of solvers used
            self.nsolvers = slvr_cfg.get('nsolvers', 4)
            self.time_begin = np.arange(self.nsolvers)*self.vtime//self.nsolvers
            self.time_end = np.arange(1,self.nsolvers+1)*self.vtime//self.nsolvers
            self.time_diff = self.time_end - self.time_begin

            #print 'time begin %s' % self.time_begin
            #print 'time end %s' % self.time_end
            #print 'time diff %s' % self.time_end

            # Create streams and events for the solvers
            self.stream = [cuda.Stream() for i in range(self.nsolvers)]

            # Create a memory pool for PyCUDA gpuarray.sum reduction results
            self.dev_mem_pool = pycuda.tools.DeviceMemoryPool()
            self.dev_mem_pool.allocate(10*ONE_KB).free()

            # Create a pinned memory pool for asynchronous transfer to GPU arrays
            self.pinned_mem_pool = pycuda.tools.PageLockedMemoryPool()
            self.pinned_mem_pool.allocate(shape=(10*ONE_KB,),dtype=np.int8).base.free()


        # Create configurations for the sub-solvers
        sub_slvr_cfg = [slvr_cfg.copy() for i in range(self.nsolvers)]
        for i, s in enumerate(sub_slvr_cfg):
            s[Options.NTIME] = self.time_diff[i]

        # Create the sub-solvers
        self.solvers = [BiroSolver(sub_slvr_cfg[i]) for i in range(self.nsolvers)]

        # Register arrays on the sub-solvers
        A_sub = copy.deepcopy(BSV2mod.A)
        P_sub = copy.deepcopy(BSV2mod.P)

        for ary in A_sub:
            # Add a transfer method
            ary['transfer_method'] = self.get_sub_transfer_method(ary['name'])
            ary['cpu'] = False
            ary['gpu'] = True

        for i, slvr in enumerate(self.solvers):
            slvr.register_properties(P_sub)
            slvr.register_arrays(A_sub)
            # Indicate that all numpy arrays on the CompositeSolver
            # have been transferred to the sub-solvers
            slvr.was_transferred = {}.fromkeys(
                [v.name for v in self._arrays.itervalues()], True)

        self.use_weight_vector = slvr_cfg.get(Options.WEIGHT_VECTOR, False)
        self.initialised = False

        self.fsm = montblanc.impl.biro.common.get_fsm(self)

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

        for r in self._arrays.itervalues():
            # Is there anything to transfer for this array?
            if not r.cpu:
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
            # Go to the start condition
            self.fsm.to_start()

            while not self.fsm.model.is_done():
                self.fsm.next()

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
