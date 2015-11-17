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

try:
    from distarray.globalapi import Context, Distribution
except ImportError as e:
    montblanc.log.error('distarray package is not installed.')
    raise

try:
    from ipyparallel import Client, CompositeError
except ImportError as e:
    montblanc.log.error('ipyparallel package is not installed.')
    raise

import collections
import copy
import numpy as np

import montblanc
import montblanc.impl.biro.v4.BiroSolver as BSV4mod
import montblanc.util as mbu

from montblanc.BaseSolver import BaseSolver
from montblanc.config import (BiroSolverConfiguration,
    BiroSolverConfigurationOptions as Options)


from remote import (query_remote_memory,
    query_remote_uuid,
    query_remote_hostname,
    create_remote_solver,
    shutdown_remote_solver)

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

        slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
        slvr_cfg[Options.VERSION] = Options.VERSION_FIVE
        slvr_cfg[Options.DATA_SOURCE] = Options.DATA_SOURCE_DEFAULTS
        slvr_cfg[Options.NBL] = self.nbl

        if hasattr(slvr_cfg, Options.MS_FILE):
            del slvr_cfg[Options.MS_FILE]

        # Import the profile
        profile = slvr_cfg.get('profile', 'mpi')
        client = Client(profile=profile)

        # Create an ipyparallel client and view
        # over the connected engines
        self.distarray_ctx = ctx = Context(client=client)
        self.valid_engines = self.get_valid_engines(self.distarray_ctx)
        print('Valid engines %s' % self.valid_engines)

        # Create the slvr variable on the remote
        ctx.view['slvr'] = None

        # Get array and property dictionaries,
        # determine memory requirements
        A, P = self.get_arys_and_props(slvr_cfg, cpu_ary_only=True)
        total_mem_required = mbu.dict_array_bytes_required(A, P)

        ctx.targets = self.valid_engines
        remote_hosts = ctx.apply(query_remote_hostname)

        print('Checking remote memory')
        ctx.targets = self.valid_engines
        mem_per_remote = ctx.apply(query_remote_memory)
        remote_mem_str = ['    engine %s: %s on %s' % (
            eid, mbu.fmt_bytes(m.available), hostname)
            for m, hostname, eid 
            in zip(mem_per_remote, remote_hosts, self.valid_engines)]
        total_remote_mem = sum([m.available for m in mem_per_remote])

        print('Memory required %s' % mbu.fmt_bytes(total_mem_required))
        print('Available memory per remote \n%s' % '\n'.join(remote_mem_str))
        print('Total available remote memory %s' %  mbu.fmt_bytes(total_remote_mem))

        if total_mem_required > total_remote_mem:
            raise ValueError(('Solving this problem requires %s '
                'memory in total. Total remote memory is %s '
                'divided into each remote engine as follows\n%s. '
                'Add more remote engines to handle this problem size.') %
                    (mbu.fmt_bytes(total_mem_required),
                    mbu.fmt_bytes(total_remote_mem),
                    '\n'.join(remote_mem_str)))

        D = DistributedBiroSolver.distribute_visibilities(
            slvr_cfg, total_mem_required,
            mem_per_remote, self.valid_engines)

        for k,v in D.iteritems():
            print('%s: %s' % (k, v))

        for engine_id, modded_dims in D.iteritems():
            print(engine_id, modded_dims)

            sub_slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
            sub_slvr_cfg.update(modded_dims)

            ctx.targets = engine_id
            res = self.distarray_ctx.apply(create_remote_solver,
                args=(sub_slvr_cfg,))

            print(res)


            #res = self.distarray_ctx.apply(shutdown_remote_solver)
            #print(res)


        import time as time
        time.sleep(2)

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

    def _distarray_context(self):
        return self.distarray_ctx


    @staticmethod
    def get_valid_engines(distarray_context):
        """
        There may be multiple engines on a single host.
        Finds engines with duplicate uuids, prunes
        them down to one engine and returns a smaller
        list of engines
        """
        ctx = distarray_context
        ctx.targets = ctx.client.ids

        # Get remote engine UUID's in order to identify
        # multiple engines on a single host
        uuids = ctx.apply(query_remote_uuid)

        print(zip(uuids, ctx.targets))

        duplicates = collections.defaultdict(list)
        for uuid, engine_id in zip(uuids, ctx.client.ids):
            duplicates[uuid].append(engine_id)

        # We only want to choose one engine per host
        return [engine_id_list[0]
            for engine_id_list
            in duplicates.itervalues()]

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

    @staticmethod
    def subdivide_visibilities(ntime, nbl, nchan, multiplier):
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

    @staticmethod
    def distribute_visibilities(slvr_cfg, total_mem_required,
        mem_per_remote, valid_engines):

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

        print('Remote memory ratios: {ratios}'.format(
            ratios=remote_mem_ratios))
        print('Cumulative remote memory ratios {ratios}'.format(
            ratios=np.cumsum(remote_mem_ratios)))

        D = {}

        for i, (r, engine_id) in enumerate(zip(remote_mem_ratios, valid_engines)):
            s_ntime, s_nbl, s_nchan = DistributedBiroSolver.subdivide_visibilities(
                ntime, nbl, nchan, r)
            s_na = 2 if s_nbl == 1 else na
            D[engine_id] = {'ntime': s_ntime, 'nbl': s_nbl, 'na': s_na, 'nchan': s_nchan}

        return D
