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

from montblanc.solvers.mb_numpy_solver import MontblancNumpySolver
from montblanc.solvers.mb_cuda_solver import MontblancCUDASolver

def copy_solver(src_slvr, dest_slvr, safe=False):
    """
    Copies arrays and properties from src_slvr to dest_slvr.
    If safe is True, complains if they don't exist on the
    destination.
    
    Primarily used for copying data from CPU Solvers to
    GPU Solvers in the test suites.
    """

    # CPU to CUDA case
    if (isinstance(src_slvr, MontblancNumpySolver) and
        isinstance(dest_slvr, MontblancCUDASolver)):
    
        sa, da = src_slvr.arrays(), dest_slvr.arrays()

        # Transfer CPU arrays to the GPU
        for a in sa.itervalues():
            if not safe and a.name not in da:
                continue

            cpu_ary = getattr(src_slvr, a.name)
            transfer_method_name = dest_slvr.transfer_method_name(a.name)
            transfer_method = getattr(dest_slvr, transfer_method_name)
            transfer_method(cpu_ary)

        sp, dp = src_slvr.properties(), dest_slvr.properties()

        # Transfer properties over
        for p in sp.itervalues():
            if not safe and p.name not in dp:
                continue

            src_prop = getattr(src_slvr, p.name)
            setattr(dest_slvr, p.name, src_prop)

    # Implement other combinations
    else:
        raise NotImplementedError('{s} to {d} copy not yet implemented'
            .format(s=type(src_slvr).__name__, d= type(dest_slvr).__name__))        
