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

from .sink_provider import SinkProvider

def gather_provider_factory(slvr, sink_provs, data_sinks):
    """ """
    target = slvr.tf_server_target
    job = slvr.tf_server_job
    task = slvr.tf_server_task
    cluster = slvr.tf_server_cluster
    schemas = slvr.hypercube.arrays()

    _validate_master_worker(cluster)

    sinks = [(n, _dtype_name(schemas[n].dtype)) for n in data_sinks]

    sink_links = [{
        "source" : ("worker", ti),
        "target" : ("master", 0),
        "sources" : sinks,
    } for ti in range(len(cluster['worker']))]

    if job == 'worker':
        return GatherSendSinkProvider(target,
            job, task, data_sinks, sink_links)
    elif job == 'master' and task == 0:
        return GatherReceiveSinkProvider(target,
            job, task, sink_provs, sink_links)
    else:
        raise ValueError("Invalid job '{}' "
                        "and task '{}'".format(job, task))

def sink_send_data_barrier(data_sink_name):
    """ Stores data from the data sink in a data barrier """
    def memoizer(self, context):
        transcoder = CubeDimensionTranscoder(a[0] for a in context.iter_args)
        descriptor = transcoder.encode(context.dimensions(copy=False))

        self._barrier.store(tuple(descriptor),
            {
                data_sink_name : context.data,
                BARRIER_KEY: descriptor,
            },
        )

    return memoizer

class GatherSendSinkProvider(BaseScatterGatherProvider, SinkProvider):
    def __init__(self, target, job, task, data_sink_names, links):
        super(GatherSendSinkProvider, self).__init__(target,
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

        self._data_sink_names = data_sink_names
        self._barrier = DataSourceBarrier(TL['names'], barrier_callback)

        # Create data sinks from list of providers,
        # wrapping them in the sink_send_data_barrier decorator
        data_sinks = { n: sink_send_data_barrier(n)
                            for n in data_sink_names }

        # Create the data sinks on this object
        for n, f in data_sinks.iteritems():
            setattr(self, n, types.MethodType(f, self))

    def signal_done(self):
        """ Indicate EOF to each target """
        TLS = self._tf_links["src_cfg"]
        self._session.run([cfg["token_put_op"] for cfg in TLS],
            feed_dict={cfg["token_ph"]: -1 for cfg in TLS})


def sink_receive_data_barrier(data_sink):
    """ Retrieves data from the data barrier for the given data sink """
    def memoizer(self, context):
        if self._barrier_closed():
            raise ValueError("GatherSendSinkProvider is done "
                            "transmitting and the data barrier "
                            "is empty.")

        transcoder = CubeDimensionTranscoder(a[0] for a in context.iter_args)
        descriptor = transcoder.encode(context.dimensions(copy=False))
        context._data = self._barrier.pop(tuple(descriptor),
                    data_sink.__name__,
                    timeout=1)

        # Invoke the data sink
        data_sink(context)

    return memoizer

class GatherReceiveSinkProvider(BaseScatterGatherProvider, SinkProvider):
    def __init__(self, target, job, task, providers, links):
        super(GatherReceiveSinkProvider, self).__init__(target,
            job, task, links)

        self._barrier = DataSinkBarrier(self._tf_links['names'])
        self._providers = providers
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
        # wrapping them in the sink_send_data_barrier decorator
        data_sinks = { n: sink_receive_data_barrier(ds)
                            for prov in self._providers
                            for n, ds in prov.sinks().iteritems() }

        # Create the data sources on this object
        for n, f in data_sinks.iteritems():
            setattr(self, n, types.MethodType(f, self))

    def _barrier_closed(self):
        return self._done_event.is_set() and len(self._barrier) == 0

