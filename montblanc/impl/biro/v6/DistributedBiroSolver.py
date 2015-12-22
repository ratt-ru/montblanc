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
    from distarray.globalapi import Context, Distribution, DistArray
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
    shutdown_remote_solver,
    create_local_ary)

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

        # Configure the dimensions of the beam cube
        self.beam_lw = self.slvr_cfg[Options.E_BEAM_WIDTH]
        self.beam_mh = self.slvr_cfg[Options.E_BEAM_HEIGHT]
        self.beam_nud = self.slvr_cfg[Options.E_BEAM_DEPTH]

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
        print('Valid engines {ve}'.format(ve=self.valid_engines))

        # Create the slvr variable on the remote
        ctx.view['slvr'] = None

        # Get array and property dictionaries
        A, P = self.get_arys_and_props(slvr_cfg, cpu_ary_only=True)

        # At this point, we can register properties
        self.register_properties(P)

        # Now get solver properties for reasoning about
        # our problem size
        SP = self.get_properties()

        # determine memory requirements
        total_mem_required = mbu.dict_array_bytes_required(A, SP)

        # Get the hostnames
        ctx.targets = self.valid_engines
        remote_hosts = ctx.apply(query_remote_hostname)

        # Get remote memory setups
        ctx.targets = self.valid_engines
        mem_per_engine = ctx.apply(query_remote_memory)
        remote_mem_str = ['    engine {e}: {b} on {h}'.format(
            e=eid, b=mbu.fmt_bytes(m.available), h=hostname)
            for m, hostname, eid 
            in zip(mem_per_engine, remote_hosts, self.valid_engines)]
        total_remote_mem = sum([m.available for m in mem_per_engine])

        print('Memory required {m}'.format(
            m=mbu.fmt_bytes(total_mem_required)))
        print('Available memory per remote \n{m}'.format(
            m='\n'.join(remote_mem_str)))
        print('Total available remote memory {m}'.format(
            m=mbu.fmt_bytes(total_remote_mem)))

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

        for dim, dim_steps in D.iteritems():
            dim_sizes = np.diff(dim_steps)
            assert len(dim_sizes) == len(self.valid_engines)

            # Create a solver on each of the remote engines,
            # modifying the appropriate dimension in each
            # solver
            for dim_size, e in zip(dim_sizes, self.valid_engines):
                ctx.targets = e
                sub_slvr_cfg = BiroSolverConfiguration(**slvr_cfg)
                sub_slvr_cfg[dim] = dim_size
                res = ctx.apply(create_remote_solver, args=(sub_slvr_cfg,))

                print('Solver creation result on engine {e}: {r}'.format(
                    e=e, r=res))

        # Get the distribution dictionary, containing array names as keys
        # and distarray distributions as values
        dists_dict = self.get_per_ary_distributions(A, SP, D)

        print('\n'.join(['{n}: {s}'.format(n=n,s=d.shape)
            for n,d in dists_dict.iteritems()]))

        # Get the ddpr dictionary, containing array names as keys and
        # distarray dim data per rank as values
        dist_ddpr_dict = DistributedBiroSolver.get_per_ary_ddpr(dists_dict)

        # Get the comm dictionary, containing array names as keys and
        # distarray MPI comms as values
        dist_comm_dict = DistributedBiroSolver.get_per_ary_comm(dists_dict)

        # Target all valid engines
        ctx.targets = self.valid_engines

        for ary in  A:
            name = ary['name']
            ddpr = dist_ddpr_dict[name]
            distribution = dists_dict[name]
            dtype = mbu.dtype_from_str(ary['dtype'], SP)

            assert len(ctx.targets) == len(distribution.localshapes())

            # Scatter the local array shape
            # to the relevant engines
            for e, s in zip(ctx.targets, distribution.localshapes()):
                ctx.view.client[e]['local_ary_shape'] = s
                ctx.view.client[e]['local_ary_dtype'] = dtype

            # Create the local arrays on the target engines
            iters_key = self.distarray_ctx.apply(create_local_ary,
                args=(distribution.comm, ddpr, name))

            # Stitch the local arrays together to create
            # the distributed array
            #dal = lambda: DistArray.from_localarrays(iters_key[0],
            #    distribution=distribution, dtype=dtype)

            dal = DistArray.from_localarrays(iters_key[0],
                distribution=distribution, dtype=dtype)

            ary['distarray_constructor'] = dal

        # Now register the arrays
        self.register_arrays(A)

        import time as time
        time.sleep(2)

    def shutdown(self):
        # And shut them down!
        ctx = self._distarray_context()
        ctx.targets = self.valid_engines
        res = self.distarray_ctx.apply(shutdown_remote_solver)
        print('Shutdown Results {r}'.format(r=res))

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

    def get_properties(self):
        # Obtain base solver property dictionary
        # and add the beam cube dimensions to it
        D = super(DistributedBiroSolver, self).get_properties()

        D.update({
            Options.E_BEAM_WIDTH : self.beam_lw,
            Options.E_BEAM_HEIGHT : self.beam_mh,
            Options.E_BEAM_DEPTH : self.beam_nud
        })

        return D

    @staticmethod
    def get_arys_and_props(slvr_cfg, cpu_ary_only=True):
        """
        Get array and property definitions
        """

        P = copy.deepcopy(BSV4mod.P)

        # Copy the v4 arrays
        if cpu_ary_only is True:
            A = [a for a in copy.deepcopy(BSV4mod.A)
                    if a['cpu'] is True]
        else:
            A = [a for a in copy.deepcopy(BSV4mod.A)]

        # On the distributed solver, just create distarrays
        for ary in A:
            ary['gpu'] = False
            ary['cpu'] = 'distarray'

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

    def get_per_ary_distributions(self, A, P, dim_distributions):
        """
        Create distarray distributions for each array in the list A.

        Returns a dictionary of distributions indexed by array name.
        """

        distributions = {}
        distarray_ctx = self._distarray_context()        
        distarray_ctx.targets = self.valid_engines

        for ary in A:
            global_dim_data = []
            # Determine the integral shape of this array
            name = ary['name']
            sshape = ary['shape']
            ishape = mbu.shape_from_str_tuple(sshape, P)

            # Iterate over both the integral and string shapes
            # If we have a pre-configured distribution for
            # this dimension, use it, otherwise just
            # take the entire dimension.
            was_distributed = False

            for isize, ssize in zip(ishape, sshape):
                if ssize in dim_distributions:
                    D = { 'dist_type' : 'b', 'bounds' : dim_distributions[ssize] }
                    was_distributed = True
                else:
                    D = { 'dist_type' : 'b', 'bounds' : [0, isize] }

                global_dim_data.append(D)

            if not was_distributed:
                global_dim_data[0] = {
                    'dist_type' : 'o',
                    'size' : ishape[0],
                    'proc_grid_size' : len(self.valid_engines)
                }


            # Create a distribution for this array
            distributions[name] = Distribution.from_global_dim_data(
                distarray_ctx, global_dim_data)

        return distributions

    @staticmethod
    def get_per_ary_ddpr(distribution_dict):
        """
        Get dim data per rank per distribution
        """

        return { n: d.get_dim_data_per_rank()
            for n, d in distribution_dict.iteritems() }

    @staticmethod
    def get_per_ary_comm(distribution_dict):
        """
        Get dim data per rank per distribution
        """

        return { n: d.comm for n, d in distribution_dict.iteritems() }

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

    # Take these methods from the v2 BiroSolver
    get_default_base_ant_pairs = \
        BSV4mod.BiroSolver.__dict__['get_default_base_ant_pairs']
    get_default_ant_pairs = \
        BSV4mod.BiroSolver.__dict__['get_default_ant_pairs']
    get_ap_idx = \
        BSV4mod.BiroSolver.__dict__['get_ap_idx']
