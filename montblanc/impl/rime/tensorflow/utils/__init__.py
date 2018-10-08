from collections import deque
from functools import wraps
from threading import Lock

_source_stack = deque()


def source_context(source):
    """
    Marks a function as associated with a particular source type.

    An internal stack

    .. code-block:: python

        @source_context("point")
        def point_body(p, coherencies):
            ...
            return p+1, coherencies

    """

    def fn_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            _source_stack.append(source)

            try:
                return fn(*args, **kwargs)
            finally:
                _source_stack.pop()

        return wrapper

    return fn_decorator


def active_source():
    try:
        return _source_stack[-1]
    except IndexError:
        raise ValueError("No active sources found")


class SingletonMixin(object):
    """
    Generic singleton mixin object
    """
    __singleton_lock = Lock()
    __singleton_instance = None

    @classmethod
    def instance(cls):
        if not cls.__singleton_instance:
            with cls.__singleton_lock:
                if not cls.__singleton_instance:
                    cls.__singleton_instance = cls()

        return cls.__singleton_instance
