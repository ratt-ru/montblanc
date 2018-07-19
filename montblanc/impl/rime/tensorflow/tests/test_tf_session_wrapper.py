import cloudpickle
import pytest

import numpy as np

from montblanc.impl.rime.tensorflow.tf_session_wrapper import (
                                            TensorflowSessionWrapper)
from montblanc.impl.rime.tensorflow.rimes.basic import (
                                            create_tf_expr as basic)

from montblanc.impl.rime.tensorflow.rimes.ddes import (
                                            create_tf_expr as ddes)


@pytest.fixture
def rime_cfg():
    return {'polarisation_type': 'linear'}


@pytest.mark.parametrize("expr", [basic, ddes])
def test_session_wrapper(expr, rime_cfg):
    w = TensorflowSessionWrapper(expr, rime_cfg)

    # Test that pickling and unpickling works
    w2 = cloudpickle.loads(cloudpickle.dumps(w))

    assert w._fn == w2._fn
    assert w._cfg == w2._cfg
    assert w._graph != w2._graph
    assert w._session != w2._session

    # Must close else test cases will hang
    w.close()
    w2.close()


@pytest.mark.parametrize("expr", [basic, ddes])
def test_session_with(expr, rime_cfg):
    with TensorflowSessionWrapper(expr, rime_cfg):
        pass


def test_session_run(rime_cfg):
    def _dummy_data(ph):
        """ Generate some dummy data given a tensorflow placeholder """
        shape = tuple(2 if s is None else s for s in ph.shape.as_list())
        return np.ones(shape, dtype=ph.dtype.as_numpy_dtype())*0.001

    with TensorflowSessionWrapper(basic, rime_cfg) as w:
        in_ds = w._datasets["inputs"]
        pt_ds = w._datasets["point_inputs"]
        pt_key = 1

        # Create some input data for the input queue and the point source map
        in_data = {n: _dummy_data(ph) for n, ph in in_ds.placeholders.items()}
        pt_data = {n: _dummy_data(ph) for n, ph in pt_ds.placeholders.items()}
        in_data['__point_keys__'] = [pt_key]

        # Insert point source data
        assert w._session.run(pt_ds.size) == 0
        w.enqueue("point_inputs", pt_key, pt_data)
        assert w._session.run(pt_ds.size) == 1

        # Insert general queue data
        assert w._session.run(in_ds.size) == 0
        w.enqueue("inputs", 100, in_data)

        # Now wait for the result
        w.dequeue(100)

        # Map is not empty, we need to manually clear it
        assert w._session.run(pt_ds.size) == 1
        w._session.run(pt_ds.clear, feed_dict={pt_ds.clear_key: [pt_key]})
        assert w._session.run(pt_ds.size) == 0
