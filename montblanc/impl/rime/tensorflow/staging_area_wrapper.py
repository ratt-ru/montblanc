from attrdict import AttrDict

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

from queue_wrapper import _get_queue_types

class StagingAreaWrapper(object):
    def __init__(self, name, fed_arrays, data_sources, shared_name=None, ordered=False):
        self._name = name
        self._fed_arrays = fed_arrays
        self._data_sources = data_sources

        # Infer types of the given fed_arrays
        self._dtypes = _get_queue_types(fed_arrays, data_sources)

        # Create placeholders for the fed arrays
        self._placeholders = placeholders = [tf.placeholder(dt,
                name="{n}_placeholder".format(n=n))
            for n, dt in zip(fed_arrays, self._dtypes)]

        self._put_key_ph = tf.placeholder(dtype=tf.int64)
        self._get_key_ph = tf.placeholder(dtype=tf.int64)
        self._peek_key_ph = tf.placeholder(dtype=tf.int64)

        self._staging_area = sa = data_flow_ops.MapStagingArea(
            self._dtypes, names=fed_arrays, ordered=ordered,
            shared_name=shared_name)

        self._put_op = sa.put(self._put_key_ph, {n: p for n, p
                                            in zip(fed_arrays, placeholders)},
                                                name="%s_put_op" % name)
        self._get_op = sa.get(self._get_key_ph, name="%s_get_op" % name)
        self._peek_op = sa.get(self._peek_key_ph, name="%s_peek_op" % name)
        self._pop_op = sa.get(name="%s_pop_op" % name)
        self._clear_op = sa.clear(name="%s_clear_op" % name)
        self._size_op = sa.size(name="%s_size_op" % name)
        self._incomplete_size_op = sa.incomplete_size(name="%s_incomplete_size_op" % name)

    @property
    def staging_area(self):
        return self._staging_area

    @property
    def fed_arrays(self):
        return self._fed_arrays

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def put_key_ph(self):
        return self._put_key_ph

    @property
    def get_key_ph(self):
        return self._get_key_ph

    @property
    def peek_key_ph(self):
        return self._peek_key_ph

    def put(self, key, data, indices=None, name=None):
        return self._staging_area.put(key, data, indices, name=name)

    def put_from_list(self, key, data, name=None):
        return self.put(key, {n: d for n,d
                                in zip(self._fed_arrays, data)},
                            name=name)

    def get(self, key=None, name=None):
        return self._staging_area.get(key, name=name)

    def peek(self, key=None, name=None):
        return self._staging_area.peek(key, name=name)

    def get_to_list(self, key=None, name=None):
        k, D = self.get(key, name=name)
        return k, [D[n] for n in self._fed_arrays]

    def get_to_attrdict(self, key=None, name=None):
        key, values = self.get(key, name=name)
        return key, AttrDict(**values)

    @property
    def put_op(self):
        return self._put_op

    @property
    def get_op(self):
        return self._get_op

    @property
    def pop_op(self):
        return self._pop_op

    @property
    def peek_op(self):
        return self._peek_op

    @property
    def clear_op(self):
        return self._clear_op

    @property
    def size_op(self):
        return self._size_op

    @property
    def incomplete_size_op(self):
        return self._incomplete_size_op

def create_staging_area_wrapper(name, fed_arrays, data_source, *args, **kwargs):
    return StagingAreaWrapper(name, fed_arrays, data_source, *args, **kwargs)
