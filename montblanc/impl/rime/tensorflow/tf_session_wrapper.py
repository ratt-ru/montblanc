from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

try:
    from cytoolz import merge
except ImportError:
    from toolz import merge

import montblanc


class TensorflowSessionWrapper(object):
    def __init__(self, fn, cfg):
        self._fn = fn
        self._cfg = cfg
        self._create_session()

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
        from montblanc.impl.rime.tensorflow.tensorflow_mock_analyser import (
            analyse_tensorflow_function,
            create_datasets)

        device_list = self._get_device_list()

        with tf.Graph().as_default():
            device = tf.DeviceSpec.from_string('/cpu:0')
            datasets, placeholders = analyse_tensorflow_function(self._fn,
                                                                 self._cfg,
                                                                 device)

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

            in_ds = dataset_info["inputs"].dataset

            # Shard the dataset over each device
            for shard, device in enumerate(device_list):
                in_ds = in_ds.shard(len(device_list), shard)
                device = tf.DeviceSpec.from_string(device.name)
                expr = self._fn(self._cfg, device, in_ds, src_maps)
                exprs.append(expr)

            global_init = tf.global_variables_initializer()

            graph.finalize()

        def _depends_on_input_ds(op):
            """ Does the supplied op depend on the input dataset? """
            for i in op.inputs:
                if (i.op.name.startswith("inputs") and
                        i.op.op_def.name == "SimpleQueueDataset"):

                    return True

            # No, recurse and check the op's inputs
            return any(_depends_on_input_ds(i.op) for i in op.inputs)

        self._inits = []
        self._closes = []

        for op in graph.get_operations():
            # Find the op responsible for initialising
            # the main dataset iterator
            if op.op_def.name == "MakeIterator" and _depends_on_input_ds(op):
                self._inits.append(op)
            # Dataset close operations
            elif op.op_def.name in ("DatasetQueueClose", "DatasetMapClose"):
                self._closes.append(op)

        # # No input dataset?
        if len(self._inits) == 0:
            raise ValueError("No input dataset iterator was created!")

        self._inits.insert(0, global_init)

        self._graph = graph
        self._session = tf.Session(graph=graph)
        self._session.run(self._inits)

    def __del__(self):
        S = getattr(self, "_session", None)

        if S is not None:
            # Run any resource close operations
            S.run(self._closes)
            # Close the session
            S.close()
            del self._session
            del self._graph
            del self._closes
            del self._inits

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self._fn, self._cfg)
