import atexit
from collections import Mapping

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

from montblanc.rime.tf_session_wrapper import TensorflowSessionWrapper


__cache_lock = Lock()
__cache = {}


def recursive_hash(d):
    if isinstance(d, (set, tuple, list)):
        return tuple((recursive_hash(e) for e in d))
    elif isinstance(d, Mapping):
        return frozenset((k, recursive_hash(v)) for k, v in d.items())
    else:
        return hash(d)


def get(fn, cfg):
    key = (hash(fn), recursive_hash(cfg))

    with __cache_lock:
        try:
            return __cache[key]
        except KeyError:
            w = TensorflowSessionWrapper(fn, cfg)
            __cache[key] = w
            return w


def clear(fn=None, cfg=None):
    if fn is None and cfg is None:
        with __cache_lock:
            for v in __cache.values():
                v.close()

            __cache.clear()
    elif fn is not None and cfg is not None:
        with __cache_lock:
            key = (hash(fn), recursive_hash(cfg))
            entry = __cache[key]
            entry.close()
            del __cache[key]
    else:
        raise ValueError("fn and cfg must both be either present or None")


atexit.register(clear)
