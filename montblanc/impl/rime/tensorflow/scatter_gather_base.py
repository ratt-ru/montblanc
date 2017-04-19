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


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

class BaseScatterGatherProvider(object):
    def __init__(self, target, job, task, links):

        self._target = target
        self._job = job
        self._task = task
        self._links = links

        self._tf_links = L = create_tensorflow_links(job, task, links)
        self._session = tf.Session(target, graph=L["graph"])

    @property
    def job(self):
        return self._job

    @property
    def task(self):
        return self._task

    def close(self):
        self._session.close()

    def name(self):
        return self.__class__.__name__


def create_tensorflow_links(job, task, links):
    with tf.Graph().as_default() as graph:
        return_dict = { 'graph' : graph }

        devspec = tf.DeviceSpec(job=job, task=task)

        with tf.device(devspec), graph.container('scatter-gather'):
            for link in links:
                source_job, source_task = link["source"]
                target_job, target_task = link["target"]
                names, dtypes = zip(*link["sources"])

                return_dict.update(names=names, dtypes=dtypes)

                tgt_devspec = tf.DeviceSpec(job=target_job,
                                            task=target_task)

                shared_sa_name = "{}_{}_scatter_data_staging_area".format(
                                    target_job, target_task)
                shared_tsa_name = "{}_{}_token_staging_area".format(
                                    target_job, target_task)

                # Reference stateful ops on the target node
                with tf.device(tgt_devspec):
                    # Staging Area that holds tokens indicating
                    # data presence in the Data Staging Area
                    token_staging_area = data_flow_ops.StagingArea(
                                        [tf.int8], shapes=[()],
                                        shared_name=shared_tsa_name)

                    # Staging Area that actually contains the data
                    data_staging_area = data_flow_ops.StagingArea(
                                        dtypes,
                                        shared_name=shared_sa_name)

                # Create put/enqueue ops on the source node
                if job == source_job and task == source_task:
                    token_ph = tf.placeholder(tf.int8)
                    token_put_op = token_staging_area.put([token_ph])

                    put_phs = [tf.placeholder(dt, name=n) for n, dt
                                in zip(names, dtypes)]

                    data_put_op = data_staging_area.put(put_phs)

                    return_dict.update(source=dict(token_ph=token_ph,
                                        token_put_op=token_put_op,
                                        put_phs=put_phs,
                                        data_put_op=data_put_op))

                # Create get/dequeue ops on the target node
                if job == target_job and task == target_task:
                    # Dequeue operation
                    token_get_op = token_staging_area.get()

                    # Indicates that there is no more data
                    # in the staging area
                    eof = tf.constant(-1, dtype=tf.int8)

                    # Dummy values
                    dummies = tuple(tf.constant(0, dtype=np.dtype(dt).type)
                        for dt in dtypes)

                    # Returns (token, dummy) if token == -1
                    # else (token, data)
                    data_get_op = tf.cond(tf.equal(token_get_op, eof),
                        lambda: (token_get_op, ) + dummies,
                        lambda: (token_get_op, ) + tuple(data_staging_area.get()))

                    data_get_op = [ data_get_op[0],
                                { n: d for n, d in zip(names, data_get_op[1:]) } ]

                    return_dict.update(target=dict(token_get_op=token_get_op,
                                                    data_get_op=data_get_op))

    return return_dict


def _validate_master_worker(cluster):
    for job in ('master', 'worker'):
        if not job in cluster:
            raise KeyError("'{}' not in "
                "cluster '{}'".format(job, cluster))

    if not len(cluster['master']) == 1:
        raise ValueError("'master' job should only contain one task. "
                         "Cluster is '{}'".format(cluster))

def _dtype_name(dtype):
    return np.dtype(dtype).name