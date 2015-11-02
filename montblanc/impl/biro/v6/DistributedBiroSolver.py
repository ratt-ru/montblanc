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

import montblanc

import montblanc.impl.biro.v4.BiroSolver as BSV4mod

from proxy_array import ProxyArray

try:
    from ipyparallel import Client, CompositeError
except ImportError as e:
    montblanc.log.error('ipyparallel package is not installed.')
    raise

from montblanc.BaseSolver import BaseSolver

class DistributedBiroSolver(BaseSolver):
    """
    Distributed Solver Implementation for BIRO
    """

    def __init__(self, slvr_cfg):
        """
        Distributed Biro Solver constructor

        Parameters:
            slvr_cfg : SolvercConfiguration
                Solver Configuration Variables
        """

        super(DistributedBiroSolver, self).__init__(slvr_cfg)

        # Copy the v4 arrays and properties and
        # modify them for use on this Solver
        A_main, P_main = \
            copy.deepcopy(BSV4mod.A), copy.deepcopy(BSV4mod.P)

        # Import the profile
        profile = slvr_cfg.get('profile', 'mpi')

        # Create an ipyparallel client and view
        # over the connected engines
        self.client = Client(profile=profile)
        self.view = self.client[:]

        from remote_handler import EngineHandler
        from montblanc.config import (BiroSolverConfiguration,
            BiroSolverConfigurationOptions as Options)

        slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
        slvr_cfg[Options.VERSION] = Options.VERSION_FIVE
        slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_DEFAULTS
        slvr_cfg[Options.NBL] = self.nbl

        if hasattr(slvr_cfg, Options.MS_FILE):
            del slvr_cfg[Options.MS_FILE]

        try:
            eh = EngineHandler(self.client, self.view)
            eh.create_remote_solvers(slvr_cfg)
        except CompositeError as e:
            e.print_traceback();

        import time as time
        time.sleep(10)

        #import numpy as np

        #ary = np.random.random(128*128)
        #proxy_array = ProxyArray(ary, self.view)

    def __enter__(self):
        """
        When entering a run-time context related to this solver,
        initialise and return it.
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        Perform shutdown when exiting a run-time context
        for this solver,
        """
        pass