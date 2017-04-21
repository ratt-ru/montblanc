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

import threading
import types

from ..scatter_gather_base import (BaseScatterGatherProvider,
    _validate_master_worker,  _dtype_name,
    BARRIER_KEY, BARRIER_DTYPE)
from ..data_source_barrier import DataSourceBarrier
from ..data_sink_barrier import DataSinkBarrier
from ..cube_dim_transcoder import CubeDimensionTranscoder

from .source_provider import SourceProvider

def scatter_provider_factory(slvr, source_provs, data_sources):
    """ """

    target = slvr.tf_server_target
    job = slvr.tf_server_job
    task = slvr.tf_server_task
    cluster = slvr.tf_server_cluster
    schemas = slvr.hypercube.arrays()

    _validate_master_worker(cluster)

    sources = [(n, _dtype_name(schemas[n].dtype)) for n in data_sources]

    source_links = [{
            "source" : ("master", 0),
            "target" : ("worker", ti),
            "sources" : sources,
        } for ti in range(len(cluster['worker']))]

    if job == 'master' and task == 0:
        return ScatterSendSourceProvider(target,
            job, task, source_provs, source_links)
    elif job == 'worker':
        return ScatterReceiveSourceProvider(target,
            job, task, data_sources, source_links)
    else:
        raise ValueError("Invalid job '{}' "
                        "and task '{}'".format(job, task))


def source_send_data_barrier(data_source):
    """ Stores data from the data source in a data barrier """
    def memoizer(self, context):
        transcoder = CubeDimensionTranscoder(a[0] for a in context.iter_args)
        descriptor = transcoder.encode(context.dimensions(copy=False))

        data = data_source(context)

        self._barrier.store(tuple(descriptor),
            {
                data_source.__name__ : data,
                BARRIER_KEY: descriptor,
            },
        )

        # TODO: Find a way to remove this, its not strictly necessary
        # On the other hand, a return doesn't really hurt
        return data

    return memoizer


class ScatterSendSourceProvider(BaseScatterGatherProvider, SourceProvider):
    def __init__(self, target, job, task, providers, links):
        super(ScatterSendSourceProvider, self).__init__(target,
            job, task, links)

        TL = self._tf_links
        TLS = TL["src_cfg"]

        # Construct a list of put ops for each target's token and data
        ops = [cfg["token_put_op"] for cfg in TLS]
        ops.extend(cfg["data_put_op"] for cfg in TLS)
        # Construct a name: placeholder map for each target
        ph_maps = [{n: ph for n, ph in zip(cfg["names"], cfg["put_phs"])}
                            for cfg in TLS]

        def barrier_callback(descriptor, entry):
            """
            Place data in the remote staging area when all
            entries are available
            """

            # Construct feed dictionary mapping data to each target's
            # placeholders
            feed_dict = {ph_map[n]: d for n, d in entry.iteritems()
                                        for ph_map in ph_maps}

            # Feed queue token
            feed_dict.update({cfg["token_ph"]: 1 for cfg in TLS})

            assert descriptor == tuple(entry[BARRIER_KEY])

            # Feed token and data
            self._session.run(ops, feed_dict)

        self._providers = providers
        self._barrier = DataSourceBarrier(TL['names'], barrier_callback)

        # Create data sources from list of providers,
        # wrapping them in the source_send_data_barrier decorator
        data_sources = { n: source_send_data_barrier(ds)
                            for prov in self._providers
                            for n, ds in prov.sources().iteritems() }

        # Create data sources on this object
        for n, f in data_sources.iteritems():
            setattr(self, n, types.MethodType(f, self))

    def signal_done(self):
        """ Indicate EOF to each target """
        TLS = self._tf_links["src_cfg"]
        self._session.run([cfg["token_put_op"] for cfg in TLS],
            feed_dict={cfg["token_ph"]: -1 for cfg in TLS})

def source_receive_data_barrier(data_source_name):
    """ Retrieves data from the data barrier for the given data source """
    def memoizer(self, context):
        if self._barrier_closed():
            raise ValueError("ScatterSendSourceProvider is done "
                            "transmitting and the data barrier "
                            "is empty.")

        transcoder = CubeDimensionTranscoder(a[0] for a in context.iter_args)
        descriptor = transcoder.encode(context.dimensions(copy=False))
        return self._barrier.pop(tuple(descriptor),
                    data_source_name,
                    timeout=1)

    return memoizer

class ScatterReceiveSourceProvider(BaseScatterGatherProvider, SourceProvider):
    def __init__(self, target, job, task, data_source_names, links):
        super(ScatterReceiveSourceProvider, self).__init__(target,
            job, task, links)

        self._barrier = DataSinkBarrier(self._tf_links['names'])
        self._data_source_names = data_source_names
        self._done_event = threading.Event()

        ops = [cfg["data_get_op"] for cfg in self._tf_links["tgt_cfg"]]

        def feed_barrier():
            """ Pull data out of staging areas and store in barrier """

            done = False

            while not done:
                for token, data in self._session.run(ops):
                    if token == -1: # Received EOF
                        done = True # last iteration
                    else:
                        barrier_key = tuple(data.pop(BARRIER_KEY))
                        self._barrier.store(barrier_key, data)

            self._done_event.set()

        # Spawn a thread to feed the barrier
        t = threading.Thread(target=feed_barrier)
        t.setDaemon(True)
        t.start()

        # Create data sources from list of providers,
        # wrapping them in the source_receive_data_barrier decorator
        data_sources = { n: source_receive_data_barrier(n)
                        for n in data_source_names }

        # Create the data sources on this object
        for n, f in data_sources.iteritems():
            setattr(self, n, types.MethodType(f, self))

    def _barrier_closed(self):
        return self._done_event.is_set() and len(self._barrier) == 0
