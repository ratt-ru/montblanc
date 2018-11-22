from collections import Mapping
from operator import itemgetter, getitem

import cloudpickle
import dask
import dask.array as da
import numpy as np
import pytest
from dask.sharedict import ShareDict
from montblanc.rime.key_pool import KeyPool
from montblanc.rime.rimes.basic import create_tf_expr as basic
from montblanc.rime.rimes.ddes import create_tf_expr as ddes
from montblanc.rime.rimes.pass_through import create_tf_expr as pass_through
from montblanc.rime.tf_session_wrapper import TensorflowSessionWrapper

from montblanc.rime.rimes.basic_multiple_sources import create_tf_expr as basic_multiple_sources


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


@pytest.mark.parametrize("iteration", xrange(100))
def test_session_run(rime_cfg, iteration):
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

        # Check that all datasets are empty
        for ds in w._datasets.values():
            assert w._session.run(ds.size) == 0


_fake_dim_chunks = {
    'point': (5, 5),
    'gaussian': (7, 7, 7),
    'sersic': (4, 4, 4),
    'row': (20, 20, 20, 20, 20),
    'time': (1, 1, 1, 1, 1),
    'chan': (8, 8),
    'corr': (4,),
    'ant': (7,),
    'beam_lw': (5,),
    'beam_mh': (5,),
    'beam_nud': (5,),
    '(u,v,w)': (3,),
    '(l,m)': (2,)
}


def output_shapes(wrapper, output_schema, chunks=False, reshapes=False):
    if not chunks and not reshapes:
        return ()

    oreshapes = []
    ochunks = []

    for o, (oname, odata) in enumerate(wrapper.placeholder_outputs.items()):
        oschema = odata['schema']
        oreshape = []
        ochunk = []

        for dim in output_schema:
            dim_chunks = _fake_dim_chunks[dim]

            try:
                oschema.index(dim)
            except ValueError:
                oreshape.append(None)
                ochunk.append((1,)*len(dim_chunks))
            else:
                oreshape.append(slice(None))
                ochunk.append(dim_chunks)

        oreshapes.append(tuple(oreshape))
        ochunks.append(tuple(ochunk))

    if chunks and reshapes:
        return (ochunks, oreshapes)
    elif chunks and not reshapes:
        return ochunks
    elif not chunks and reshapes:
        return oreshapes
    else:
        raise ValueError("Logic Error")


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


def _key_from_dsn(source_dataset_name):
    if not source_dataset_name.endswith("_inputs"):
        raise ValueError("Source Dataset name %s did not "
                         "end with '_inputs'")

    return "__" + source_dataset_name[:-len("_inputs")] + "_keys__"


_key_pool = KeyPool()


def _rime_factory(wrapper, output_schema):
    # Establish a sorted sequence of inputs that will correspond
    # to the arguments in the factory function
    phs = wrapper.placeholders.copy()

    main_phs = phs.pop("inputs")
    main_inputs = list(sorted(main_phs.keys()))

    source_inputs = {dsn: (_key_from_dsn(dsn), list(sorted(sphs.keys())))
                     for dsn, sphs in phs.items()}

    oreshapes = output_shapes(wrapper, output_schema, reshapes=True)

    def _rime(*args):
        main_args = args[0:len(main_inputs)]
        main_feed = {}
        main_key = _key_pool.get(1)
        source_keys = []

        dequeue_dict = {"inputs": main_key[0]}

        key_lists = []
        start = end = len(main_inputs)

        # Determine keys for our source inputs
        for dsn, (source_key, inputs) in source_inputs.items():
            # Extract argument range for this source type
            end += len(inputs)
            ds_args = args[start:end]

            if not all(isinstance(a, type(ds_args[0])) for a in ds_args[1:]):
                raise TypeError("Argument types were not all the same "
                                "type for dataset %s" % dsn)

            if isinstance(ds_args[0], list):
                nentries = len(ds_args[0])

                if not all(nentries == len(a) for a in ds_args[1:]):
                    raise ValueError("Expected lists of the same length")

                main_feed[source_key] = keys = _key_pool.get(nentries)
            elif isinstance(ds_args[0], np.ndarray):
                main_feed[source_key] = keys = _key_pool.get(1)
            else:
                raise ValueError("Unhandled input type '%s'"
                                 % type(ds_args[0]))

            key_lists.append(keys)
            source_keys.extend(keys)
            dequeue_dict[dsn] = keys
            start = end

        inputs = {n: a for n, a in zip(main_inputs, main_args)}
        inputs["time_index"].fill(0)
        inputs["antenna1"][:] = 0
        inputs["antenna2"][:] = 1

        main_feed.update(inputs)
        print("Enqueueing main inputs %s" % main_key[0])
        wrapper.enqueue("inputs", main_key[0], main_feed)
        print("Enqueueing main inputs %s done" % main_key[0])

        start = end = len(main_inputs)

        # Iteration producing something like
        # "point_inputs", ("__point_keys__", ["point_lm", "point_stokes"])
        for (dsn, (_, inputs)), keys in zip(source_inputs.items(), key_lists):
            # Extract argument range for this source type
            end += len(inputs)
            ds_args = args[start:end]

            print("Enqueueing %s inputs %s" % (dsn, keys))

            # Handle lists of source chunks
            if isinstance(ds_args[0], list):
                for e, k in enumerate(keys):
                    wrapper.enqueue(dsn, k, {n: a[e] for n, a
                                             in zip(inputs, ds_args)})
            # Handle a single source chunk
            elif isinstance(ds_args[0], np.ndarray):
                wrapper.enqueue(dsn, keys[0], {n: a for n, a
                                               in zip(inputs, ds_args)})
            else:
                raise ValueError("Unhandled input type '%s'"
                                 % type(ds_args[0]))

            print("Enqueueing %s inputs %s done" % (dsn, keys))

            start = end

        res = wrapper.dequeue(dequeue_dict)
        _key_pool.release(source_keys)
        _key_pool.release(main_key)

        # Return data, reshaping into shapes that dask will understand
        return tuple(out[r] for out, r in zip(res, oreshapes))

    return _rime


def _fake_dask_inputs(wrapper):
    phs = wrapper.placeholders.copy()

    main_phs = phs.pop("inputs")
    ordered_inputs = list(sorted(main_phs.items(), key=itemgetter(0)))

    for dsn, dphs in phs.items():
        ordered_inputs.extend(sorted(dphs.items(), key=itemgetter(0)))

    dask_inputs = []

    for input_name, ph_data in ordered_inputs:
        chunks = tuple((s,) if isinstance(s, int)
                       else _fake_dim_chunks[s]
                       for s in ph_data['schema'])
        shape = tuple(map(sum, chunks))
        dtype = ph_data['type'].as_numpy_dtype()

        # Create random data
        array = da.random.random(size=shape, chunks=chunks).astype(dtype)*0.001
        # We associate time chunks with row chunks
        schema = tuple("row" if a == "time" else a for a in ph_data['schema'])

        dask_inputs.append((input_name, schema, array))

    return dask_inputs


@pytest.mark.parametrize("expr", [basic_multiple_sources])
@pytest.mark.parametrize("iteration", range(1))
def test_dask_wrap(expr, rime_cfg, iteration):
    with TensorflowSessionWrapper(expr, rime_cfg) as w:
        # We're always producing this kind of output
        output_schema = ["row", "chan", "corr"]

        rime_fn = _rime_factory(w, output_schema)
        dask_inputs = _fake_dask_inputs(w)

        token = dask.base.tokenize(*(a for _, _, a in dask_inputs))
        rime_name = "rime-" + token

        name_schemas = [(a.name, s) for _, s, a in dask_inputs]
        numblocks = {a.name: a.numblocks for _, _, a in dask_inputs}

        # Create the graph from all the inputs
        rime_dsk = da.core.top(rime_fn, rime_name, output_schema,
                               *(a for pair in name_schemas for a in pair),
                               numblocks=numblocks)

        # Remove the need to recurse into input lists within rime_fn
        rime_dsk = _flatten_singletons(rime_dsk)

        outputs = []
        ochunks = output_shapes(w, output_schema, chunks=True)

        # Create graphs for each of the outputs produced by rime_fn
        for o, (oname, odata) in enumerate(w.placeholder_outputs.items()):
            # Create the dask graph
            dsk = ShareDict()
            dsk.update(rime_dsk)

            # Add input dask graphs
            for _, _, a in dask_inputs:
                dsk.update(a.__dask_graph__())

            # Extract individual tuple components produced by the
            # rime function.
            out_name = oname + "-" + token
            get_dsk = {(out_name,) + key[1:]: (getitem, key, o)
                       for key in rime_dsk.keys()}

            dsk.update(get_dsk)

            dtype = odata['type'].as_numpy_dtype()
            output = da.Array(dsk, out_name, ochunks[o], dtype=dtype)
            outputs.append(output)

        # Test that compute works
        for output in outputs:
            assert output.compute().shape == output.shape

        # Test that all keys have been released from the pool
        assert _key_pool.all_released() is True

        # Check that all datasets are empty
        for ds in w._datasets.values():
            assert w._session.run(ds.size) == 0


@pytest.mark.skip
@pytest.mark.parametrize("expr", [pass_through])
@pytest.mark.parametrize("iteration", range(10))
def test_determinism(expr, iteration, rime_cfg):
    with TensorflowSessionWrapper(expr, rime_cfg) as w:
        # We're always producing this kind of output
        output_schema = ["row", "chan", "corr"]

        rime_fn = _rime_factory(w, output_schema)
        dask_inputs = _fake_dask_inputs(w)

        token = dask.base.tokenize(*(a for _, _, a in dask_inputs))
        rime_name = "rime-" + token

        name_schemas = [(a.name, s) for _, s, a in dask_inputs]
        numblocks = {a.name: a.numblocks for _, _, a in dask_inputs}

        # Create the graph from all the inputs
        rime_dsk = da.core.top(rime_fn, rime_name, output_schema,
                               *(a for pair in name_schemas for a in pair),
                               numblocks=numblocks)

        # Remove the need to recurse into input lists within rime_fn
        rime_dsk = _flatten_singletons(rime_dsk)

        outputs = []
        ochunks = output_shapes(w, output_schema, chunks=True)

        # Create graphs for each of the outputs produced by rime_fn
        for o, (oname, odata) in enumerate(w.placeholder_outputs.items()):
            # Create the dask graph
            dsk = ShareDict()
            dsk.update(rime_dsk)

            # Add input dask graphs
            for _, _, a in dask_inputs:
                dsk.update(a.__dask_graph__())

            # Extract individual tuple components produced by the
            # rime function.
            out_name = oname + "-" + token
            get_dsk = {(out_name,) + key[1:]: (getitem, key, o)
                       for key in rime_dsk.keys()}

            dsk.update(get_dsk)

            dtype = odata['type'].as_numpy_dtype()
            output = da.Array(dsk, out_name, ochunks[o], dtype=dtype)
            outputs.append(output)

        # Test that compute works
        for output in outputs:
            assert output.compute().shape == output.shape

        # Test that all keys have been released from the pool
        assert _key_pool.all_released() is True

        # Check that all datasets are empty
        for ds in w._datasets.values():
            assert w._session.run(ds.size) == 0
