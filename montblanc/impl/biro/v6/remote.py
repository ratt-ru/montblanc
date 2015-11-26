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
        return str(traceback.format_exc())

def shutdown_remote_solver():
    """
    Shutdown the solver on the remote.
    """
    import traceback

    try:
        global slvr
        slvr.shutdown()
        del slvr

        return "Shutdown succeeded"
    except Exception as e:
        return str(traceback.format_exc())

def stitch_local_arys(comm, ddpr, A):
    """
    Create distributed arrays by wrapping solver arrays
    on the remote
    """

    import traceback

    try:
        from distarray.localapi import LocalArray
        from distarray.localapi.maps import Distribution

        dim_data = () if len(ddpr) == 0 else ddpr[comm.Get_rank()]

        ldist = Distribution(comm=comm, dim_data=dim_data)

        global slvr
    except Exception as e:
        return str(traceback.format_exc())
