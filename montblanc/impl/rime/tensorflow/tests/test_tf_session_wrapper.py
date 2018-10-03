from collections import Mapping

import cloudpickle
import dask
import dask.array as da
from dask.sharedict import ShareDict
import numpy as np
import pytest

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
    with TensorflowSessionWrapper(expr, rime_cfg) as w:
        # Test that pickling and unpickling works
        with cloudpickle.loads(cloudpickle.dumps(w)) as w2:
            assert w._fn == w2._fn
            assert w._cfg == w2._cfg
            assert w._graph != w2._graph
            assert w._session != w2._session


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
        w.dequeue({"inputs": 100, "point_inputs": [pt_key]})

        # Check that input queue + map is clear
        assert w._session.run(in_ds.size) == 0
        assert w._session.run(pt_ds.size) == 0


def _rime_factory(inputs):
    try:
        data_index = inputs.index("data")
    except IndexError:
        raise ValueError("This rime function depends on the use "
                         "of the 'data' input within the rime function ")

    def _rime(*args):
        return args[data_index]

    return _rime


_fake_dim_chunks = {
    'source': (5, 5, 5),
    'row': (20, 20, 20, 20, 20),
    'time': (1, 1, 1, 1, 1),
    'chan': (16,),
    'corr': (4,),
    'ant': (7,),
    '(u,v,w)': (3,),
    '(l,m)': (2,)
}


def _fake_dask_inputs(input_data):
    dask_inputs = []

    for name, data in input_data:
        chunks = tuple(_fake_dim_chunks[s] for s in data['schema'])
        shape = tuple(map(sum, chunks))
        dtype = data['type'].as_numpy_dtype()

        array = da.random.random(size=shape, chunks=chunks).astype(dtype)
        schema = tuple("row" if a == "time" else a for a in data['schema'])

        dask_inputs.append((name, schema, array))

    return dask_inputs


def output_chunks(output_schema):
    return tuple(_fake_dim_chunks[s] for s in output_schema)


def _flatten_singletons(D):
    """ Recursively simplify tuples and list of length 1 """

    # lists and tuples should remain lists and tuples
    if isinstance(D, list):
        return (_flatten_singletons(D[0]) if len(D) == 1
                else [_flatten_singletons(v) for v in D])
    elif isinstance(D, tuple):
        return (_flatten_singletons(D[0]) if len(D) == 1
                else tuple(_flatten_singletons(v) for v in D))
    elif isinstance(D, Mapping):
        return {k: _flatten_singletons(v) for k, v in D.items()}
    else:
        return D


def test_dask_wrap(rime_cfg):
    with TensorflowSessionWrapper(basic, rime_cfg) as w:
        inputs = []

        for dsn, ds in w.placeholders.items():
            inputs.extend(ds.items())

        outputs = tuple((k, v['schema']) for k, v
                        in w.placeholder_outputs.items())

        inputs = sorted(inputs)
        output_schema = max(outputs, key=lambda o: len(o[1]))[1]
        # We're always producing this kind of output
        output_schema = ["row", "chan", "corr"]

        rime_fn = _rime_factory([name for name, _ in inputs])

        dask_inputs = _fake_dask_inputs(inputs)

        token = dask.base.tokenize(*(a for _, _, a in dask_inputs))
        rime_name = "rime-" + token

        name_schemas = [(a.name, s) for _, s, a in dask_inputs]
        numblocks = {a.name: a.numblocks for _, _, a in dask_inputs}

        rime_dsk = da.core.top(rime_fn, rime_name, output_schema,
                               *(a for pair in name_schemas for a in pair),
                               numblocks=numblocks)

        rime_dsk = _flatten_singletons(rime_dsk)

        dsk = ShareDict()
        dsk.update(rime_dsk)

        for _, _, a in dask_inputs:
            dsk.update(a.__dask_graph__())

        output = da.Array(dsk, rime_name,
                          output_chunks(output_schema),
                          dtype=np.complex128)

        assert output.compute().shape == output.shape
