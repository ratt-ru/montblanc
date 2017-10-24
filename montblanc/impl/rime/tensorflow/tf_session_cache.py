import atexit
from collections import defaultdict
from contextlib import contextmanager
import logging

import six

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

class TensorflowSessionCache(object):
    def __init__(self):
        self.refcount = defaultdict(lambda: 0)
        self.cache = {}
        self.lock = Lock()

    @contextmanager
    def open(self, myopen, *args, **kwargs):
        # TODO(sjperkins). Use myopen callable as a unique identifier in the cache key
        # This fails in the distributed case at present as the same callable will have
        # a different ID in the same graph on the same worker.
        #key = (myopen,) + (args,) + (frozenset(kwargs.items()),)
        key = (args,) + (frozenset(kwargs.items()),)
        with self.lock:
            try:
                session = self.cache[key]
            except KeyError:
                session = myopen(*args, **kwargs)
                self.cache[key] = session

            self.refcount[key] += 1

        try:
            yield session
        finally:
            with self.lock:
                self.refcount[key] -= 1

    def size(self):
        with self.lock:
            return len(self.cache)

    def clear(self):
        with self.lock:
            for key, session in six.iteritems(self.cache):
                try:
                    session.close()
                except AttributeError:
                    log.warn("Unable to call 'close()' on key '%s'" % key)

            self.cache.clear()
            self.refcount.clear()

__TF_SESSION_CACHE = TensorflowSessionCache()

def tf_session_cache():
    global __TF_SESSION_CACHE
    return __TF_SESSION_CACHE

# Clear the session cache on exit
def __clear_session_cache():
    global __TF_SESSION_CACHE
    __TF_SESSION_CACHE.clear()

atexit.register(__clear_session_cache)
