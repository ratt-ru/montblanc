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

import atexit
import collections
import copy
import numpy as np
import random
import string
import sys
import types

import montblanc
import montblanc.impl.biro.v4.BiroSolver as BSV4mod
import montblanc.util as mbu

from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

def _remote_check_memory(slvr_cfg):
    """ Query CPU memory available on the remote engine """
    import psutil
    return psutil.virtual_memory()

def _remote_get_uuid():
    """
    Get an identifier for the remote.
    Uses uuid.getnode() to get something
    based on the MAC address.
    """
    import uuid
    return uuid.getnode()

def _remote_create_solver(slvr_cfg):
    """
    Create a solver on the remote, using the
    supplied Solver Configuration
    """
    global slvr
    import montblanc.factory
    slvr = montblanc.factory.rime_solver(slvr_cfg)
    slvr.initialise()
    return str(slvr)

class EngineHandler(object):
    def __init__(self, client, dview):
        self.remote_id = randomword(20)
        self.client = client
        self.dview = dview

        with self.dview.sync_imports():
            #import numpy as np
            import montblanc
            import montblanc.factory

        #self.dview['slvr'] = {}

    @staticmethod
    def get_arys_and_props(slvr_cfg, cpu_ary_only=True):
        """
        Get array and property definitions
        """

        P = slvr_cfg.copy()

        # Copy the v4 arrays 
        if cpu_ary_only is True:
            A = [a for a in copy.deepcopy(BSV4mod.A)
                    if a['cpu'] is True]
        else:
            A = [a for a in copy.deepcopy(BSV4mod.A)]

        # Duplication of functionality in BaseSolver constructor
        P['ft'], P['ct'] = mbu.float_dtypes_from_str(P[Options.DTYPE])
        src_nr_vars = mbu.sources_to_nr_vars(slvr_cfg[Options.SOURCES])
        # Sum to get the total number of sources
        nsrc = sum(src_nr_vars.itervalues())
        P.update(src_nr_vars)
        P['nsrc'] = nsrc

        return A, P

    def get_valid_engines(self):
        """
        There may be multiple engines on a single host.
        Finds engines with duplicate uuids, prunes
        them down to one engine and returns a smaller
        list of engines
        """
        # Target all engines
        self.dview.targets = self.client.ids

        # Get remote engine UUID's in order to identify
        # multiple engines on a single host
        uuids = self.dview.apply_sync(_remote_get_uuid)

        duplicates = collections.defaultdict(list)
        for uuid, engine_id in zip(uuids, self.client.ids):
            duplicates[uuid].append(engine_id)

        # We only want to choose one engine per host
        return [engine_id_list[0]
            for engine_id_list
            in duplicates.itervalues()]

    @staticmethod
    def sub_vis(ntime, nbl, nchan, multiplier):
        # Determine visibilities and number of reduced visibilities
        nvis = ntime*nbl*nchan
        nreducedvis = int(nvis*multiplier)

        # Result array holding final [ntime, nbl, nchan] values
        result = [1, 1, 1]
        # Enumerate over these dimensions
        dims = [nchan, nbl, ntime]
        cumprod = 1

        # Iterate over the three dimensions, cumulatively
        # multiplying the product
        for i, dim in enumerate(dims):
            prev_cumprod, cumprod = cumprod, cumprod*dim
            # The cumulative product still matched the
            # reduced visibilities
            # use the full dimension
            if nreducedvis >= cumprod:
                result[i] = dim
            # They don't fit
            else:
                result[i] = nreducedvis / prev_cumprod
                break

        return tuple(reversed(result))

    def distribute_visibilities(self, slvr_cfg, total_mem_required, mem_per_remote):
        ntime = slvr_cfg[Options.NTIME]
        nbl = slvr_cfg[Options.NBL]
        na = slvr_cfg[Options.NA]
        nchan = slvr_cfg[Options.NCHAN]
        nvis = ntime*nbl*nchan

        mem_vis_ratio = total_mem_required / float(nvis)
        available_remote_mem = [m.available for m in mem_per_remote]
        total_remote_mem = sum(available_remote_mem)
        remote_mem_ratios = [m / float(total_remote_mem)
            for m in available_remote_mem]

        D = {}

        for i, (r, engine_id) in enumerate(zip(remote_mem_ratios, self.valid_engines)):
            s_ntime, s_nbl, s_nchan = self.sub_vis(ntime, nbl, nchan, r)
            s_na = 2 if s_nbl == 1 else na
            D[engine_id] = {'ntime': s_ntime, 'nbl': s_nbl, 'na': s_na, 'nchan': s_nchan}

        return D

    def create_remote_solvers(self, slvr_cfg):
        # Find  out which engines we wish to use and target them
        self.valid_engines = valid_engines = self.get_valid_engines()
        self.dview.targets = valid_engines

        print('Valid engines %s' % valid_engines)

        # Get array and property dictionaries,
        # determine memory requirements
        A, P = self.get_arys_and_props(slvr_cfg, cpu_ary_only=True)
        total_mem_required = mbu.dict_array_bytes_required(A, P)

        print('Checking remote memory')
        mem = self.dview.apply_sync(_remote_check_memory, slvr_cfg)
        mem_per_remote = ['    engine %s: %s' % (eid, mbu.fmt_bytes(m.available))
            for m, eid in zip(mem, valid_engines)]
        total_remote_mem = sum([m.available for m in mem])

        print('Memory required %s' % mbu.fmt_bytes(total_mem_required))
        print('Available memory per remote \n%s' % '\n'.join(mem_per_remote))
        print('Total available remote memory %s' %  mbu.fmt_bytes(total_remote_mem))

        if total_mem_required > total_remote_mem:
            mem_per_remote = ['engine %s: %s' % \
                (eid, mbu.fmt_bytes(m.available))
                for m, eid in zip(mem, valid_engines)]

            raise ValueError(('Solving this problem requires %s '
                'memory in total. Total remote memory is %s '
                'divided into each remote engine as follows %s. '
                'Add more remote engines to handle this problem size.') %
                    (mbu.fmt_bytes(total_mem_required),
                    mbu.fmt_bytes(total_remote_mem),
                    mem_per_remote))

        D = self.distribute_visibilities(slvr_cfg, total_mem_required, mem)

        for engine_id, modded_dims in D.iteritems():
            sub_slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
            sub_slvr_cfg.update(modded_dims)

            self.dview.targets = engine_id
            print('Created remote solver on engine %s' % engine_id)
            res = self.dview.apply_sync(_remote_create_solver, sub_slvr_cfg)

            print(res)
        """
        for (viable, modded_dims), engine_id in zip(res, valid_engines):
            sub_slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
            sub_slvr_cfg.update(modded_dims)

            print('Creating remote solvers')
            self.dview.targets = engine_id
            res = self.dview.apply_sync(_remote_create_solver, sub_slvr_cfg)
            print(res[0])
        """

        self.dview.targets = self.client.ids