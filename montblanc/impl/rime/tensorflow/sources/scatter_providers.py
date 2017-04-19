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
            job, task, source_provs, source_links)
    else:
        raise ValueError("Invalid job '{}' "
                        "and task '{}'".format(job, task))


def source_send_data_barrier(data_source):
    """ Stores data from the data source in a data barrier """
    def memoizer(self, context):
        transcoder = CubeDimensionTranscoder(a[0] for a in context.iter_args)
        descriptor = transcoder.encode(context.dimensions(copy=False))

        self._barrier.store(tuple(descriptor),
            {
                data_source.__name__ : data_source(context),
                BARRIER_KEY: descriptor,
            },
        )

    return memoizer


class ScatterSendSourceProvider(BaseScatterGatherProvider, SourceProvider):
    def __init__(self, target, job, task, providers, links):
        super(ScatterSendSourceProvider, self).__init__(target,
            job, task, links)

        TL = self._tf_links

        print TL.keys(), target, job, task
        TLS = TL["source"]
        ph_map = { n: ph for n, ph in zip(TL["names"], TLS["put_phs"]) }

        def barrier_callback(descriptor, entry):
            """
            Place data in the remote staging area when all
            entries are available
            """

            # Feed data placeholders with data
            feed_dict = { ph_map[n]: d for n, d in entry.iteritems() }
            # Feed queue token
            feed_dict.update({TLS["token_ph"]: 1})

            assert descriptor == tuple(entry[BARRIER_KEY])

            # Feed token and data
            self._session.run([TLS["token_put_op"], TLS["data_put_op"]],
                feed_dict)

        self._providers = providers
        self._barrier = DataSourceBarrier(self._tf_links['names'],
                barrier_callback)

        # Create data sources from list of providers,
        # wrapping them in the source_send_data_barrier decorator
        data_sources = { n: source_send_data_barrier(ds)
                            for prov in self._providers
                            for n, ds in prov.sources().iteritems() }

        # Create the data sources on this object
        for n, f in data_sources.iteritems():
            setattr(self, n, types.MethodType(f, self))

    def signal_done(self):
        TLS = self._tf_links["source"]
        self._session.run(TLS["token_put_op"],
            feed_dict={ TLS["token_ph"] : -1})

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
                    timeout=None)

    return memoizer

class ScatterReceiveSourceProvider(BaseScatterGatherProvider, SourceProvider):
    def __init__(self, target, job, task, data_source_names, links):
        super(ScatterReceiveSourceProvider, self).__init__(target,
            job, task, links)

        self._barrier = DataSinkBarrier(self._tf_links['names'])
        self._data_source_names = data_source_names
        self._done_event = threading.Event()

        data_get_op = self._tf_links["target"]["data_get_op"]

        def feed_barrier():
            """ Pull data out of staging areas and store in barrier """

            while True:
                token, data = self._session.run(data_get_op)

                if token == -1:
                    self._done_event.set()
                    break

                barrier_key = tuple(data.pop(BARRIER_KEY))
                self._barrier.store(barrier_key, data)

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
