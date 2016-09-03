import numpy as np
import tensorflow as tf
import pyrap.tables as pt
import sys
import types

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
    def __init__(self, name, queue_size, fed_arrays, data_sources):
        self._name = name
        self._fed_arrays = fed_arrays
        self._data_sources = data_sources

        # Infer types of the given fed_arrays
        self._queue_types = _get_queue_types(fed_arrays, data_sources)

        # Create placeholders for the fed arrays
        self._placeholders = [tf.placeholder(dt, name="{n}_placeholder".format(n=n))
            for n, dt in zip(fed_arrays, self._queue_types)]

        # Create a FIFOQueue of a given size with the supplied queue types
        self._queue = tf.FIFOQueue(queue_size, self._queue_types, name=name)

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

def create_queue_wrapper(name, queue_size, fed_arrays, data_sources):
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
    return QueueWrapper(name, queue_size, fed_arrays, data_sources)