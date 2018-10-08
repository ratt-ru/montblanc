from montblanc.impl.rime.tensorflow.utils import (active_source,
                                                  source_decorator)


def test_source_decorator():
    @source_decorator("point")
    def fn(a, b):
        assert active_source() == "point"
        return a + b

    assert fn(2, 3) == 5

    @source_decorator("gaussian")
    def fn(a, b):
        assert active_source() == "gaussian"
        return a + b

    assert fn(2, 3) == 5

    @source_decorator("point")
    def fn(a, b):
        @source_decorator("gaussian")
        def gaussian_fn(a, b):
            assert active_source() == "gaussian"
            return a + b

        assert active_source() == "point"
        return gaussian_fn(a, b)

    assert fn(2, 3) == 5
