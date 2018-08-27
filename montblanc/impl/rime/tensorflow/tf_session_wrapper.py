from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread

from dask.sizeof import sizeof, getsizeof
import numpy as np
import tensorflow as tf

try:
    from cytoolz import merge
except ImportError:
    from toolz import merge

import montblanc
from montblanc.impl.rime.tensorflow.map_dataset import TensorMap
from montblanc.impl.rime.tensorflow.tensorflow_mock_analyser import (
    analyse_tensorflow_function,
    create_datasets,
    MapDatasetInfo,
    QueueDatasetInfo)


def _requires_input_ds(op):
    """ Does the supplied op depend on the input dataset? """
    for i in op.inputs:
        if (i.op.name.startswith("shard_") and
                i.op.name.endswith("/inputs") and
                i.op.op_def.name == "SimpleQueueDataset"):

            return True

    # No, recurse and check the op's inputs
    return any(_requires_input_ds(i.op) for i in op.inputs)


class TensorflowSessionWrapper(object):
    def __init__(self, fn, cfg):
        self._fn = fn
        self._cfg = cfg
        self._create_session()

        self._eval_thread = Thread(target=self.evaluate_expr)
        self._eval_thread.setDaemon(True)
        self._eval_thread.start()

    def _get_device_list(self):
        """ Get a list of the preferred devices """
        import tensorflow as tf

        try:
            requested_device = self._cfg["device_type"]
        except KeyError:
            requested_device = "GPU"

        with tf.Session() as S:
            tf_device_list = S.list_devices()
            device_map = {d.device_type: [] for d in tf_device_list}

            for d in tf_device_list:
                device_map[d.device_type].append(d)

        try:
            device_list = device_map[requested_device]
        except KeyError:
            montblanc.log.info("Couldn't find any %s devices. "
                               "Reverting to CPU." % requested_device)
            try:
                device_list = device_map["CPU"]
            except KeyError:
                raise ValueError("No CPU devices where found")

        if len(device_list) == 0:
            raise ValueError("No devices found %s" % device_map)

        return device_list

    def _create_session(self):
        """ Create a tensorflow session """
        import tensorflow as tf
        from tensorflow.contrib.framework import nest

        device_list = self._get_device_list()

        with tf.Graph().as_default():
            datasets, placeholders, outputs = analyse_tensorflow_function(
                                                                self._fn,
                                                                self._cfg,
                                                                'fake')

        # Add in a chunk_key uniquely identifying the chunk of data
        datasets["inputs"].variables()["chunk_key"]
        placeholders["inputs"]["chunk_key"] = {
            'allowed_types': [tf.int64],
            'default': tf.int64,
            'default_type_name': 'int64',
            'type': tf.int64,
            'ops': [],
            'schema': (),
        }

        # Extract the main input dataset definitions
        input_ds = {"inputs": datasets.pop("inputs")}

        with tf.Graph().as_default() as graph:
            # Now create source datasets composed of maps
            # and main input dataset composed of a queue
            src_ds = create_datasets(datasets, placeholders, "map")
            input_ds = create_datasets(input_ds, placeholders, "queue")

            dataset_info = merge(input_ds, src_ds)
            src_maps = {ds_name: ds.tensor_map for ds_name, ds
                        in src_ds.items()}

            # Create an expression for each device
            exprs = []
            key_idx = []

            # Get the main input dataset
            in_ds = dataset_info["inputs"].dataset

            output_map = TensorMap(tuple(o['type'] for _, o in outputs))
            self._output_map_pop_key = tf.placeholder(tf.int64)
            self._output_map_pop = output_map.pop(self._output_map_pop_key)

            # Shard the dataset over each device
            for shard, device in enumerate(device_list):
                in_ds = in_ds.shard(len(device_list), shard)

                out_types = in_ds.output_types
                out_types = nest.flatten_with_joined_string_paths(out_types)

                # Identify the chunk key
                # This could get dodgy at some point
                key_idx.append([i for i, (n, t) in enumerate(out_types)
                               if n == "chunk_key"][0])

                device = tf.DeviceSpec.from_string(device.name)

                with tf.name_scope("shard_%s" % shard):
                    expr = self._fn(self._cfg, device, in_ds, src_maps)

                exprs.append(expr)

            shard_it_keys = [None] * len(device_list)
            close_ops = ("DatasetQueueClose", "DatasetMapClose")

            self._iterator_inits = []
            self._closes = []

            for op in graph.get_operations():
                # Find the op responsible for initialising
                # the main dataset iterator
                if op.op_def.name == "MakeIterator" and _requires_input_ds(op):
                    self._iterator_inits.append(op)
                # Dataset close operations
                elif op.op_def.name in close_ops:
                    self._closes.append(op)
                # Iterator gets, get the chunk_key output tensor
                elif op.op_def.name.endswith("GetNext"):
                    op_str = op.name.split('/')

                    if len(op_str) == 2 and op_str[-1].endswith("GetNext"):
                        scope, op_name = op_str
                        chunk_key_i = key_idx[shard]
                        shard_it_keys[int(scope[-1])] = op.outputs[chunk_key_i]

            assert all(ik is not None for ik in shard_it_keys)

            # # No input dataset?
            if len(self._iterator_inits) == 0:
                raise ValueError("No input dataset iterator was created!")

            map_inserts = []

            for key, expr in zip(shard_it_keys, exprs):
                map_inserts.append(output_map.insert(key, expr))

            self._global_init = tf.global_variables_initializer()

            graph.finalize()

        self._datasets = dataset_info
        self._exprs = map_inserts
        self._keys = shard_it_keys

        self._graph = graph
        self._session = tf.Session(graph=graph)

        # Run initialisation
        self._session.run([self._global_init, self._iterator_inits])

    def enqueue(self, dataset, key, data):
        """ Enqueue ``data`` with ``key`` in the specified ``dataset`` """
        try:
            ds = self._datasets[dataset]
        except KeyError:
            raise ValueError("Unknown dataset %s. "
                             "Valid datasets %s" %
                             (dataset, self._datasets.keys()))

        ph = ds.placeholders
        feed_dict = {ph[k]: v for k, v in data.items()}

        if isinstance(ds, QueueDatasetInfo):
            if not dataset == "inputs":
                raise ValueError("Must be inputs dataset")

            feed_dict[ph["chunk_key"]] = key
        elif isinstance(ds, MapDatasetInfo):
            feed_dict[ds.put_key] = key

        self._session.run([ds.put], feed_dict=feed_dict)

    def dequeue(self, keys):
        if isinstance(keys, (int, np.integer)):
            return self._session.run(self._output_map_pop, feed_dict={
                                     self._output_map_pop_key: keys})
        elif isinstance(keys, dict):
            # Retrieve results from the inputs dataset first
            # TODO(sjperkins)
            # Running this concurrently with the map dataset clears below
            # can produce conditions where required input gets cleared
            # before tensorflow can retrieve it for execution.
            # It would be nice to use tf.control_dependencies
            # and submit it all in one session run
            try:
                ds_keys = keys.pop("inputs")
            except KeyError:
                raise ValueError("No inputs dataset")
            else:
                if isinstance(ds_keys, (int, np.integer)):
                    res = self._session.run(self._output_map_pop, feed_dict={
                                            self._output_map_pop_key: ds_keys})
                else:
                    raise ValueError("Queue key %s must be "
                                     "scalar integer" % (ds_keys,))

            # Now clear out the input datasets
            ops = []
            feed_dict = {}

            for dataset, ds_keys in keys.items():
                try:
                    ds = self._datasets[dataset]
                except KeyError:
                    raise ValueError("Unknown dataset %s. "
                                     "Valid datasets %s" %
                                     (dataset, self._datasets.keys()))

                if isinstance(ds, QueueDatasetInfo):
                    raise ValueError("Only inputs queue allowed")
                elif isinstance(ds, MapDatasetInfo):
                    ops.append(ds.clear)
                    feed_dict[ds.clear_key] = ds_keys
                else:
                    raise ValueError("Invalid dataset type")

            self._session.run(ops, feed_dict=feed_dict)

        return res

    def evaluate_expr(self):
        while True:
            try:
                self._session.run(self._exprs)
            except tf.errors.OutOfRangeError:
                # Try run each of the key expression pairs
                # individually to fully clear the entries out
                for e in self._exprs:
                    try:
                        self._session.run(e)
                    except tf.errors.OutOfRangeError:
                        pass

                break

    def close(self):
        if not self._session._closed:
            # Close all queues/maps
            self._session.run(self._closes)
            # Wait for the evaluation thread to join
            self._eval_thread.join()
            # Close the session
            self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()
        return etype is None

    def __del__(self):
        self.close()

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self._fn, self._cfg)


@sizeof.register(TensorflowSessionWrapper)
def sizeof_tf_session_wrapper(o):
    """ Size derived from function and config dictionary *only* """
    return getsizeof(o._fn) + getsizeof(o._cfg)
