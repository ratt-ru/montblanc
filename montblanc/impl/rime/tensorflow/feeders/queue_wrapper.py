import numpy as np
import tensorflow as tf
import pyrap.tables as pt
import types

def _get_queue_types(fed_arrays, data_sources):
    """
    Given a dictionaries of supplied and default arrays,
    return a list of types associated with each array in fed_arrays.
    """

    # Preferably use supplied data else take from defaults
    try:
        return [data_sources.get(n)[1] for n in fed_arrays]
    except KeyError as e:
        raise ValueError("Array '{k}' was not provided in either "
            "the 'supplied' or 'defaults' arrays".format(k=e.message))

class QueueWrapper(object):
    def __init__(self, queue_size, fed_arrays, data_sources):
        self._fed_arrays = fed_arrays
        self._data_sources = data_sources

        # Infer types of the given fed_arrays
        self._queue_types = _get_queue_types(fed_arrays, data_sources)

        # Create placeholders for the fed arrays
        self._placeholders = [tf.placeholder(dt, name="{n}_placeholder".format(n=n))
            for n, dt in zip(fed_arrays, self._queue_types)]

        # Create a FIFOQueue of a given size with the supplied queue types
        self._queue = tf.FIFOQueue(queue_size, self._queue_types)

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

    def placeholder_enqueue_op(self):
        """ Return a placeholder op for injecting into the Graph """
        return self.queue.enqueue(self.placeholders)

    def dequeue_op(self):
        return self.queue.dequeue()

    def __str__(self):
        return 'Queue of size {s} with with types {t}'.format(
            s=len(self._queue_types),
            t=self._queue_types)

def create_queue_wrapper(queue_size, fed_arrays, data_sources):
    """
    Arguments
        fed_arrays: list
            array names that will be fed by this queue
        data_sources: dict
            (lambda/method, dtype) tuples, keyed on array names

    """
    return QueueWrapper(queue_size, fed_arrays, data_sources)