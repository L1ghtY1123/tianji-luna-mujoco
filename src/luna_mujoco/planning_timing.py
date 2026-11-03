"""Request-local, inclusive wall timings; never synchronize or alter SDK work."""
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_current = ContextVar('planning_timings', default=None)


@contextmanager
def collect_timings():
    timings = {}
    token = _current.set(timings)
    try:
        yield timings
    finally:
        _current.reset(token)


@contextmanager
def timed_stage(name):
    timings = _current.get()
    if timings is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        entry = timings.setdefault(name, {'calls': 0, 'wall_s': 0.})
        entry['calls'] += 1
        entry['wall_s'] += time.perf_counter() - started


def measured(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with timed_stage(name):
                return function(*args, **kwargs)
        return wrapped
    return decorate
