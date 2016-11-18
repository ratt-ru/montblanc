import collections
import itertools
import sys

from attrdict import AttrDict
import tensorflow as tf

def _get_queue_types(fed_arrays, data_sources):
    """
    Given a list of arrays to feed in fed_arrays, return
    a list of associated queue types, obtained from tuples
    in the data_sources dictionary
    """
    try:
        return [data_sources[n].dtype for n in fed_arrays]
    except KeyError as e:
        raise ValueError("Array '{k}' has no data source!"
            .format(k=e.message)), None, sys.exc_info()[2]

class QueueWrapper(object):
    def __init__(self, name, queue_size, fed_arrays, data_sources, shared_name=None):
        self._name = name
        self._fed_arrays = fed_arrays
        self._data_sources = data_sources

        # Infer types of the given fed_arrays
        self._queue_types = _get_queue_types(fed_arrays, data_sources)

        # Create placeholders for the fed arrays
        self._placeholders = [tf.placeholder(dt, name="{n}_placeholder".format(n=n))
            for n, dt in zip(fed_arrays, self._queue_types)]

        # Create a FIFOQueue of a given size with the supplied queue types
        self._queue = tf.FIFOQueue(queue_size,
            self._queue_types, name=name, shared_name=shared_name)

        # Create enqueue operation using placeholders
        self._enqueue_op = self._queue.enqueue(self.placeholders)

        self._dequeue_op = self._queue.dequeue()

        self._close_op = self._queue.close()

        self._size_op = self._queue.size()

    @property
    def name(self):
        return self._name

    @property
    def queue_types(self):
        return self._queue_types

    @property
    def placeholders(self):
        return self._placeholders

    @property
    def queue(self):
        return self._queue

    @property
    def fed_arrays(self):
        return self._fed_arrays

    @property
    def enqueue_op(self):
        return self._enqueue_op

    @property
    def dequeue_op(self):
        return self._dequeue_op

    def dequeue(self):
        return self._queue.dequeue()

    def dequeue_to_dict(self):
        return {k: v for k, v in itertools.izip(
            self._fed_arrays, self._queue.dequeue())}

    def dequeue_to_attrdict(self):
        return AttrDict((k, v) for k, v in itertools.izip(
            self._fed_arrays, self._queue.dequeue()))

    @property
    def close_op(self):
        return self._close_op

    def close(self):
        return self._queue.close()

    @property
    def size_op(self):
        return self._size_op

    def size(self):
        return self._queue.size()

    def __str__(self):
        return 'Queue of size {s} with with types {t}'.format(
            s=len(self._queue_types),
            t=self._queue_types)

class SingleInputMultiQueueWrapper(QueueWrapper):
    def __init__(self, name, queue_size, fed_arrays, data_sources,
        shared_name=None, count=4):

        super(SingleInputMultiQueueWrapper, self).__init__(name, queue_size,
            fed_arrays, data_sources, shared_name)

        R = range(1, count)

        extra_names = ['%s_%s' % (name, i) for i in R]
        extra_shared_names = ['%s_%s' % (shared_name, i) for i in R]

        extra_queues = [tf.FIFOQueue(queue_size, self._queue_types,
                name=n, shared_name=sn)
            for n, sn in zip(extra_names, extra_shared_names)]

        extra_enqueue_ops = [q.enqueue(self._placeholders) for q in extra_queues]
        extra_dequeue_ops = [q.dequeue() for q in extra_queues]
        extra_close_ops = [q.close() for q in extra_queues]
        extra_size_ops = [q.size() for q in extra_queues]

        self._names = [self._name] + extra_names
        self._queues = [self._queue] + extra_queues
        self._enqueue_ops = [self._enqueue_op] + extra_enqueue_ops
        self._dequeue_ops = [self._dequeue_op] + extra_dequeue_ops
        self._size_ops = [self._size_op] + extra_size_ops

    @property
    def name(self):
        return self._names

    @property
    def queue(self):
        return self._queues

    @property
    def enqueue_op(self):
        return self._enqueue_ops

    @property
    def dequeue_op(self):
        return self._dequeue_ops

    @property
    def close_op(self):
        return self._close_ops


def create_queue_wrapper(name, queue_size, fed_arrays, data_sources, *args, **kwargs):
    """
    Arguments
        name: string
            Name of the queue
        queue_size: integer
            Size of the queue
        fed_arrays: list
            array names that will be fed by this queue
        data_sources: dict
            (lambda/method, dtype) tuples, keyed on array names

    """

    qtype = SingleInputMultiQueueWrapper if 'count' in kwargs else QueueWrapper
    return qtype(name, queue_size, fed_arrays, data_sources, *args, **kwargs)

