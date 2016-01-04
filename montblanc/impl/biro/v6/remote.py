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

def query_remote_memory():
    """ Query CPU memory available on the remote engine """
    import psutil
    return psutil.virtual_memory()

def query_remote_uuid():
    """
    Get an identifier for the remote.
    Uses uuid.getnode() to get something
    based on the MAC address.
    """
    import uuid
    return uuid.getnode()

def query_remote_hostname():
    """
    Get the remote hostname
    """
    import socket
    return socket.gethostname()    

def create_remote_solver(slvr_cfg):
    """
    Create a solver on the remote, using the
    supplied Solver Configuration
    """

    import traceback

    try:
        import montblanc.factory

        global slvr

        slvr = montblanc.factory.rime_solver(slvr_cfg)
        slvr.initialise()
        return str(slvr)
    except Exception as e:
        raise Exception(str(traceback.format_exc()))

def create_local_ary(comm, ddpr, name):
    """
    Create distributed array by wrapping solver arrays
    on the remote.

    Requires that local_ary_shape and local_ary_dtype
    are scattered to this engine.

    # Scatter the local array shape
    # to the relevant engines
    >>> for e, s in zip(ctx.targets, distribution.localshapes()):
    >>>    ctx.view.client[e]['local_ary_shape'] = s
    >>>    ctx.view.client[e]['local_ary_dtype'] = dtype
    """

    import traceback

    try:
        import numpy as np
        from distarray.localapi import LocalArray
        from distarray.localapi.maps import Distribution
        import montblanc.util as mbu

        global slvr

        if slvr is None:
            raise ValueError('Solver does not exist!')

        # Attempt to find the appropriate array on
        # this remote solver
        cpu_name = mbu.cpu_name(name)
        cpu_ary = getattr(slvr, cpu_name, None)

        if cpu_ary is None:
            raise AttributeError((
                'slvr.{n} was not present '
                'when creating a LocalArray as part of '
                'a DistributedArray.').format(
                    n=cpu_name))

        if cpu_ary.shape != local_ary_shape:
            raise ValueError((
                'The shape, {ashape}, of slvr.{n} does not '
                'match the supplied shape {sshape} '
                'when creating a LocalArray as part of '
                'a DistributedArray.').format(
                    n=cpu_name,
                    ashape=cpu_ary.shape,
                    sshape=local_ary_shape))

        # Create the LocalArray on this remote, and return
        # a proxy to it. Used for creating
        dim_data = () if len(ddpr) == 0 else ddpr[comm.Get_rank()]
        ldist = Distribution(comm=comm, dim_data=dim_data)
        res = LocalArray(ldist, buf=cpu_ary)
        return proxyize(res)

    except Exception as e:
        raise Exception(str(traceback.format_exc()))

def shutdown_remote_solver():
    """
    Shutdown the solver on the remote.
    """
    import traceback

    try:
        global slvr
        
        if slvr is None:
            raise ValueError('slvr is None!')

        slvr.shutdown()
        del slvr

        return "Shutdown succeeded"
    except Exception as e:
        raise Exception(str(traceback.format_exc()))

def remote_solve():
    """
    Call solve() on the remote solvers
    """

    import traceback

    try:
        global slvr

        if slvr is None:
            raise ValueError('slvr is None!')

        slvr.solve()

        return slvr.X2_cpu
    except Exception as e:
        raise Exception(str(traceback.format_exc()))
