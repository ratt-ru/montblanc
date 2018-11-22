import pytest

from montblanc.impl.rime.tensorflow.tf_session_cache import (get as cache_get,
                                                             recursive_hash)
from montblanc.impl.rime.tensorflow.rimes.basic_multiple_sources import (
                                create_tf_expr as basic_multiple_sources)


@pytest.fixture
def rime_cfg():
    return {'polarisation_type': 'linear'}


def test_session_cache(rime_cfg):
    w = cache_get(basic_multiple_sources, rime_cfg)
    w2 = cache_get(basic_multiple_sources, rime_cfg)

    assert w == w2


def test_recursive_hash():
    h = recursive_hash({'foo': 'bar',
                        'v': 1,
                        'pluge': {'qux': 'corge'}})

    h2 = recursive_hash({'foo': 'bar',
                         'v': 1,
                         'pluge': {'qux': 'corge'}})

    assert h == h2
