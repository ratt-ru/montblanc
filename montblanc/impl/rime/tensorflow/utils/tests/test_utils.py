import pytest

from montblanc.impl.rime.tensorflow.utils import (active_source,
                                                  source_context)


def test_source_context():
    @source_context("point")
    def fn(a, b):
        assert active_source() == "point"
        return a + b

    assert fn(2, 3) == 5

    @source_context("gaussian")
    def fn(a, b):
        assert active_source() == "gaussian"
        return a + b

    assert fn(2, 3) == 5

    @source_context("point")
    def fn(a, b):
        @source_context("gaussian")
        def gaussian_fn(a, b):
            assert active_source() == "gaussian"
            return a + b

        assert active_source() == "point"
        return gaussian_fn(a, b)

    assert fn(2, 3) == 5

    with pytest.raises(ValueError) as e:
        active_source()

    assert "No active sources found" in e.value.message
