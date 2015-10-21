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

import montblanc

try:
    from ipyparallel import Client
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

        # Import the profile
        profile = slvr_cfg.get('profile', None)

        # Create an ipyparallel client and view
        # over the connected engines
        self.client = Client(profile=profile)
        self.view = self.client[:]
