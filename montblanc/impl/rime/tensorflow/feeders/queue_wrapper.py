import numpy as np
import tensorflow as tf
import pyrap.tables as pt
import types

def _get_queue_types(fed_arrays, defaults, supplied):
    """
    Given a dictionaries of supplied and default arrays,
    return a list of types associated with each array in fed_arrays.
    """

    def _get_queue_type(array_data):
        """ Return a list of types for each array in fed_array """

        head, tail = ((array_data[0], array_data[1:])
            if isinstance(array_data, (tuple, list))
            else (array_data, None) )

        # Supplied with some type in the head
        if isinstance(head, type):
            return head
        # CASA Measurement Set, get the type
        # of the given column name in the tail
        if isinstance(head, pt.table):
            column = tail[0]
            col_desc = head.getcoldesc(column)
            return MS_TO_NP_TYPE_MAP[col_desc['valueType']]
        # Tensorflow tensor
        elif isinstance(head, tf.python.framework.ops.Tensor):
            return head.dtype
        # Numpy array
        elif isinstance(head, np.ndarray):
            return head.dtype
        # Functor
        elif isinstance(head, (types.LambdaType, types.MethodType)):
            return tail[0]
        else:
            raise ValueError("Unhandled queue type {qt}".format(qt=type(head)))

    # Preferably use supplied data else take from defaults
    try:
        array_data = (supplied.get(n) if n in supplied
            else defaults.get(n) for n in fed_arrays)
        return [_get_queue_type(a) for a in array_data]
    except KeyError as e:
        raise ValueError("Array '{k}' was not provided in either "
            "the 'supplied' or 'defaults' arrays".format(k=e.message))


class QueueWrapper(object):
    def __init__(self, queue_size, fed_arrays, defaults, supplied):
        self._defaults = defaults
        self._fed_arrays = fed_arrays
        self._supplied = supplied

        # Infer types of the given fed_arrays
        self._queue_types = _get_queue_types(fed_arrays, defaults, supplied)

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
    def value_names(self):
        return self._fed_arrays

    def placeholder_enqueue_op(self):
        """ Return a placeholder op for injecting into the Graph """
        return self.queue.enqueue(self.placeholders)

    def __str__(self):
        return 'Queue of size {s} with with types {t}'.format(
            s=len(self._queue_types),
            t=self._queue_types)

def create_queue_wrapper(queue_size, fed_arrays, defaults, supplied):
    """
    Arguments
        fed_arrays: list
            array names that will be fed by this queue
        supplied: dict
            numpy arrays/functions/MStables used to feed arrays named
            in fed_arrays
        defaults:
            defaults taken from this dictionary if necessary

    """
    return QueueWrapper(queue_size, fed_arrays, defaults, supplied)

"""
fed_arrays = ['observed_vis', 'alpha', 'stokes', 'lm', 'flag', 'weight']

ms = pt.table('/home/sperkins/data/WSRT.MS', ack=False)

defaults = {
    'alpha' : (lambda x: np.zeros(1000), np.float32),
    'flag' : tf.Variable(tf.zeros((1000,), dtype=tf.uint8)),
    'weight' : (ms, 'WEIGHT'),
}

supplied = {
    'observed_vis': np.empty((10,10)),
    'stokes': np.empty((10,10)),
    'lm' : np.empty((100,100))
}

print create_queue_wrapper(10, fed_arrays, defaults,  supplied)
ms.close()
"""