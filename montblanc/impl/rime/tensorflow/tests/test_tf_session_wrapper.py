import cloudpickle
import pytest

import numpy as np

from montblanc.impl.rime.tensorflow.tf_session_wrapper import (
                                            TensorflowSessionWrapper)
from montblanc.impl.rime.tensorflow.rimes.basic import (
                                            create_tf_expr as basic)

from montblanc.impl.rime.tensorflow.rimes.ddes import (
                                            create_tf_expr as ddes)


@pytest.mark.parametrize("expr", [basic, ddes])
def test_session_wrapper(expr):
    cfg = {'polarisation_type': 'linear'}
    w = TensorflowSessionWrapper(expr, cfg)

    # Test that pickling and unpickling works
    w2 = cloudpickle.loads(cloudpickle.dumps(w))

    assert w._fn == w2._fn
    assert w._cfg == w2._cfg
    assert w._graph != w2._graph
    assert w._session != w2._session


@pytest.mark.parametrize("expr", [basic, ddes])
def test_session_with(expr):
    cfg = {'polarisation_type': 'linear'}

    with TensorflowSessionWrapper(expr, cfg):
        pass

