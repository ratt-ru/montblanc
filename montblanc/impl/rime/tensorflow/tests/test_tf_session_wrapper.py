from montblanc.impl.rime.tensorflow.tf_session_wrapper import (
                                            TensorflowSessionWrapper)
from montblanc.impl.rime.tensorflow.rimes.basic import (
                                            create_tf_expr)


import cloudpickle


def test_session_wrapper():
    cfg = {'polarisation_type': 'linear'}
    w = TensorflowSessionWrapper(create_tf_expr, cfg)

    # Test that pickling and unpickling works
    w2 = cloudpickle.loads(cloudpickle.dumps(w))

    assert w._fn == w2._fn
    assert w._cfg == w2._cfg
    assert w._graph != w2._graph
    assert w._session != w2._session
