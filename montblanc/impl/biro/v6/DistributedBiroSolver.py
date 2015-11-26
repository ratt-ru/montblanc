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

        # Import the profile and create the
        # ipyparallel client
        profile = slvr_cfg.get('profile', 'mpi')
        client = Client(profile=profile)

        # Create a distarray context, in turn based
        # on an ipyparallel client
        self.distarray_ctx = ctx = Context(client=client)
        self.valid_engines = self.get_valid_engines(self.distarray_ctx)
        print('Valid engines %s' % self.valid_engines)

        # Create the slvr variable on the remote
        ctx.view['slvr'] = None

        # Get array and property dictionaries,
        # determine memory requirements
        A, P = self.get_arys_and_props(slvr_cfg, cpu_ary_only=True)
        total_mem_required = mbu.dict_array_bytes_required(A, P)

        # Get the hostnames
        ctx.targets = self.valid_engines
        remote_hosts = ctx.apply(query_remote_hostname)

        # Get remote memory setups
        print('Checking remote memory')
        ctx.targets = self.valid_engines
        mem_per_engine = ctx.apply(query_remote_memory)
        remote_mem_str = ['    engine %s: %s on %s' % (
            eid, mbu.fmt_bytes(m.available), hostname)
            for m, hostname, eid 
            in zip(mem_per_engine, remote_hosts, self.valid_engines)]
        total_remote_mem = sum([m.available for m in mem_per_engine])

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


        D = DistributedBiroSolver.get_subdivided_dims(slvr_cfg,
            total_remote_mem, mem_per_engine, self.valid_engines)

        assert len(D) == 1

        print('Subdivided dimensions {d}'.format(d=D))

        dists = self.get_per_ary_distributions(A, P, D,
            self._distarray_context())        

        # This isn't really a for loop, there's only 1 element in D
        for dim_name, subdivision in D.iteritems():
            print('{n}: {v}'.format(n=dim_name, v=subdivision))

            S = np.diff(subdivision)

            assert len(S) == len(self.valid_engines)

            # Start the solver
            for dim_size, engine_id in zip(S, self.valid_engines):
                print('Setting {n} to {v} on engine {e}'.format(
                    n=dim_name, v=dim_size, e=engine_id))
                sub_slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
                sub_slvr_cfg[dim_name] = dim_size

                ctx.targets = engine_id
                res = self.distarray_ctx.apply(create_remote_solver,
                    args=(sub_slvr_cfg,))
                print(res)


        # And shut them down!
        for engine_id in self.valid_engines:
            ctx.targets = engine_id
            res = self.distarray_ctx.apply(shutdown_remote_solver)
            print(res)

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
    def get_subdivided_dims(slvr_cfg, total_remote_mem,
            mem_per_engine, valid_engines):

        ntime = slvr_cfg[Options.NTIME]
        nbl = slvr_cfg[Options.NBL]
        na = slvr_cfg[Options.NA]
        nchan = slvr_cfg[Options.NCHAN]

        # Determine available memory ratio's per engine
        mem = np.array([m.available for m in mem_per_engine])
        ratios = mem / float(total_remote_mem)

        def distribute_dim(ratios, dimsize):
            """
            Distribute dimensions according to supplied ratios,
            using a heuristic
            """
            # Divide the dimension up by the ratios
            # Then use around to find the closest integer
            # The idea here is that if an engine doesn't have
            # enough memory around will drop the amount of the
            # dimension assigned to it to zero, making the
            # problem invalid
            division = dimsize*ratios
            distribution = np.around(division).astype(np.int64).tolist()

            # Found some zeros. Bail out
            if np.any(distribution == 0):
                return False, distribution

            dist_sum = sum(distribution)

            # Massage values in the largest dimensions if dist_sum != dimsize
            if dimsize > dist_sum:
                for i in np.flipud(np.argsort(distribution)):
                    if  dimsize <= dist_sum:
                        break

                    dist_sum += 1
                    distribution[i] += 1
            elif dimsize < dist_sum:
                for i in np.flipud(np.argsort(distribution)):
                    if  dimsize >= dist_sum:
                        break

                    dist_sum -= 1
                    distribution[i] -= 1

            """
            print('Ratios {r}'.format(r=ratios))
            print('Distribution {d}'.format(d=distribution))
            print('Distribution sum {s} dimsize {d}'.format(
                s=dist_sum, d=dimsize))
            """

            if dist_sum != dimsize:
                return False, distribution

            return True, distribution

        # Try distribute along the time dimension first
        valid, distribution = distribute_dim(ratios, ntime)

        if valid:
            return { Options.NTIME: np.insert(np.cumsum(distribution),0,0) }

        valid, distribution = distribute_dim(ratios, nbl)

        if valid:
            return { Options.NBL: np.insert(np.cumsum(distribution),0,0) }

        raise ValueError('Subdivision of the problem by '
            'time (ntime) or baseline (nbl) is not possible')

    @staticmethod
    def get_per_ary_distributions(A, P, dim_distributions, distarray_ctx):
        """
        Create distarray distributions for each array in the list A.

        Returns a dictionary of distributions indexed by array name.
        """

        distributions = {}

        for ary in A:
            global_dim_data = []
            # Determine the integral shape of this array
            ishape = mbu.shape_from_str_tuple(ary['shape'], P)

            # Iterate over both the integral and string shapes
            # If we have a pre-configured distribution for
            # this dimension, use it, otherwise just
            # take the entire dimension.
            for isize, ssize in zip(ishape, ary['shape']):
                if ssize in dim_distributions:
                    D = { 'dist_type' : 'b', 'bounds' : dim_distributions[ssize] }
                else:
                    D = { 'dist_type' : 'b', 'bounds' : [0, isize] }

                global_dim_data.append(D)

            # Create a distribution for this array
            distributions[ary['name']] = Distribution.from_global_dim_data(
                distarray_ctx, global_dim_data)

        return distributions

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