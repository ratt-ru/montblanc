from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from cytoolz import merge
except ImportError:
    from toolz import merge


class TensorflowSessionWrapper(object):
    def __init__(self, fn, cfg):
        self._fn = fn
        self._cfg = cfg
        self._create_session()

    def _create_session(self):
        import tensorflow as tf
        from montblanc.impl.rime.tensorflow.tensorflow_mock_analyser import (
            analyse_tensorflow_function,
            create_datasets)

        with tf.Session() as S:
            device_list = S.list_devices()

        with tf.Graph().as_default() as fake_graph:
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

        # Find the op responsible for initialising the main dataset iterator
        input_init_op = [op for op in graph.get_operations()
                         if op.op_def.name == "MakeIterator"
                         and _depends_on_input_ds(op)]

        # No input dataset?
        if len(input_init_op) == 0:
            raise ValueError("No input dataset iterator was created!")

        self._inits = [global_init] + input_init_op

        # Dataset close operations
        self._closes = [op for op in graph.get_operations()
                        if op.op_def.name
                        in ("DatasetMapClose", "DatasetQueueClose")]

        self._graph = graph
        self._session = tf.Session(graph=graph)
        self._session.run(self._inits)

    def __setstate__(self, args):
        self.__init__(*args)

    def __getstate__(self):
        return (self._fn, self._cfg)
