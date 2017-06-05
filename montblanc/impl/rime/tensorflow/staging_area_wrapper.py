from attrdict import AttrDict

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

from queue_wrapper import _get_queue_types

class StagingAreaWrapper(object):
    def __init__(self, name, fed_arrays, data_sources, shared_name=None):
        self._name = name
        self._fed_arrays = fed_arrays
        self._data_sources = data_sources

        # Infer types of the given fed_arrays
        self._dtypes = _get_queue_types(fed_arrays, data_sources)

        # Create placeholders for the fed arrays
        self._placeholders = placeholders = [tf.placeholder(dt,
                name="{n}_placeholder".format(n=n))
            for n, dt in zip(fed_arrays, self._dtypes)]

        self._staging_area = sa = data_flow_ops.StagingArea(self._dtypes,
            names=fed_arrays, shared_name=shared_name)

        self._put_op = sa.put({n: p for n, p in zip(fed_arrays, placeholders)})
        self._get_op = sa.get()

    @property
    def staging_area(self):
        return self._staging_area

    @property
    def fed_arrays(self):
        return self._fed_arrays

    @property
    def placeholders(self):
        return self._placeholders

    def put(self, data):
        return self._staging_area.put(data)

    def put_from_list(self, data):
        return self.put({n: d for n,d in zip(self._fed_arrays, data)})

    def get(self):
        return self._staging_area.get()

    def get_to_list(self):
        D = self.get()
        return [D[n] for n in self._fed_arrays]

    def get_to_attrdict(self):
        return AttrDict(**self.get())

    @property
    def put_op(self):
        return self._put_op

    @property
    def get_op(self):
        return self._get_op


def create_staging_area_wrapper(name, fed_arrays, data_source, *args, **kwargs):
    return StagingAreaWrapper(name, fed_arrays, data_source, *args, **kwargs)
