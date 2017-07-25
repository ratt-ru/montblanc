from __future__ import print_function

import collections
from pprint import pprint

import attr
import dask
import dask.array as da
import distributed as dd
import hypercube
import numpy as np

import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow.RimeSolver import (
    _partition,
    _setup_hypercube)


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("scheduler_address")
    return parser

def create_hypercube():
    cube = hypercube.HyperCube()
    _setup_hypercube(cube, montblanc.rime_solver_cfg())
    cube.update_dimension("npsrc", global_size=10, lower_extent=0, upper_extent=2)
    cube.update_dimension("nsrc", global_size=10, lower_extent=0, upper_extent=2)
    cube.update_dimension("ntime", global_size=100, lower_extent=0, upper_extent=10)
    cube.update_dimension("nbl", global_size=10, lower_extent=0, upper_extent=5)
    cube.update_dimension("nchan", global_size=64, lower_extent=0, upper_extent=64)
    return cube

if __name__ == "__main__":
    args = create_argparser().parse_args()

    with dd.Client(args.scheduler_address) as client:
        client.restart()

        # Create a hypercube for setting up our dask arrays
        cube = create_hypercube()
        print(cube)

        # Take all arrays flagged as input
        iter_dims = ['ntime', 'nbl']
        input_arrays = { a.name: a for a in cube.arrays().itervalues()
                                         if 'input' in a.tags }

        def _setup_worker(dask_worker=None):
            """ Setup a thread local store and a thread lock on each worker """
            import threading

            import tensorflow as tf

            slvr_cfg = {'polarisation_type' : 'linear'}

            from montblanc.impl.rime.tensorflow.key_pool import KeyPool

            def _setup_tensorflow():
                from montblanc.impl.rime.tensorflow.RimeSolver import (
                    _construct_tensorflow_staging_areas,
                    _construct_tensorflow_expression)

                from tensorflow.python.client import device_lib
                devices = device_lib.list_local_devices()

                with tf.Graph().as_default() as compute_graph:
                    # Create our data feeding structure containing
                    # input/output staging_areas and feed once variables
                    feed_data = _construct_tensorflow_staging_areas(
                        cube, iter_dims,
                        [d.name for d in devices])

                    # Construct tensorflow expressions for each device
                    exprs = [_construct_tensorflow_expression(feed_data, slvr_cfg, dev, d)
                        for d, dev in enumerate([d.name for d in devices])]

                    # Initialisation operation
                    init_op = tf.global_variables_initializer()
                    # Now forbid modification of the graph
                    compute_graph.finalize()

                session = tf.Session("", graph=compute_graph)
                session.run(init_op)

                TensorflowConfig = attr.make_class("TensorflowConfig",
                                        ["session", "feed_data", "exprs"])

                return TensorflowConfig(session, feed_data, exprs)

            dask_worker._worker_lock = threading.Lock()
            dask_worker.tf_cfg = _setup_tensorflow()
            dask_worker.key_pool = KeyPool()

            return "OK"

        assert all([v == "OK" for v in client.run(_setup_worker).values()])

        sched_info = client.scheduler_info()

        nr_master=1
        nr_worker=len(sched_info["workers"])-1

        src_data_sources, feed_many, feed_once = _partition(iter_dims,
                                                            input_arrays)

        feed_once = { a.name: a for a in feed_once }
        feed_many = { a.name: a for a in feed_many }

        fo = feed_once.keys()
        fm = feed_many.keys()

        def _create_dask_arrays(cube):
            """ Create dask arrays """
            def _create_dask_array(array):
                size = cube.dim_global_size(*array.shape)
                chunks = tuple(cube.dim_extent_size(*array.shape, single=False))
                name = '-'.join((array.name, dask.base.tokenize(array.name)))
                A = da.ones(shape=size, chunks=chunks, dtype=array.dtype, name=name)
                return A

            def _check_arrays_size(arrays):
                maximum = 4*1024*1024*1024
                total_bytes = sum(a.nbytes for a in arrays.values())
                #print("Total Size", mbu.fmt_bytes(total_bytes))

                if total_bytes >= maximum:
                    raise ValueError("%s greater than %s, quitting " % (
                                        mbu.fmt_bytes(total_bytes),
                                        mbu.fmt_bytes(maximum)))

            arrays = { n: _create_dask_array(a) for n, a in input_arrays.items() }
            _check_arrays_size(arrays)
            return arrays

        D = _create_dask_arrays(cube)
        #D = { n: client.persist(v) for n,v in D.items() }

        pprint(D)

        Klass = attr.make_class("Klass", D.keys())

        def _predict(*args, **kwargs):
            w = dd.get_worker()

            tf_cfg = w.tf_cfg
            session = tf_cfg.session
            local_cpu = tf_cfg.feed_data.local_cpu
            feed_internal = local_cpu.feed_internal
            feed_once = local_cpu.feed_once
            feed_many = local_cpu.feed_many
            feed_sources = tf_cfg.feed_data.local_cpu.sources
            exprs = tf_cfg.exprs
            key_pool = w.key_pool

            print("Feed Sources {}".format({ k: v.fed_arrays for k, v
                                             in feed_sources.iteritems() }))

            K = Klass(*args)
            D = attr.asdict(K)

            def _display(k, v):
                if isinstance(v, np.ndarray):
                    return "ndarray{}".format(v.shape,)
                elif isinstance(v, collections.Sequence):
                    return "sequence[{}]".format(len(v))
                else:
                    return v

            pprint({ k: _display(k, v) for k, v in D.items() })

            def _source_keys_and_feed_fn(k, sa):
                """ Returns (keys, feed function) for given source staging area """

                # arrays in the staging area to feed
                arrays = { n: (getattr(K, n), ph) for n, ph
                                    in zip(sa.fed_arrays, sa.placeholders) }
                # Get the actual arrays
                data = [t[0] for t in arrays.values()]

                if not all(type(data[0]) == type(d) for d in data):
                    raise ValueError("Type mismatch in arrays "
                                     "supplied for {}".format(k))

                # Handle single ndarray case
                if isinstance(data[0], np.ndarray):
                    print("Handling numpy arrays for {}".format(k))
                    if data[0].nbytes == 0:
                        print("{} is zero-length, ignoring".format(k))
                        return [], lambda: None

                    keys = key_pool.get(1)
                    feed_dict = {ph: d for n, (d, ph) in arrays.items()}
                    feed_dict[sa.put_key_ph] = keys[0]
                    from functools import partial
                    fn = partial(session.run, sa.put_op, feed_dict=feed_dict)
                    return keys, fn

                # Handle multiple ndarrays in a list case
                elif isinstance(data[0], list):
                    print("Handling list of size {} for {}".format(len(data[0]), k))
                    keys = key_pool.get(len(data[0]))

                    def fn():
                        for i, k in enumerate(keys):
                            feed_dict = { ph: d[i] for n, (d, ph) in arrays.items() }
                            feed_dict[sa.put_key_ph] = k
                            session.run(sa.put_op, feed_dict=feed_dict)

                    return keys, fn

                raise ValueError("Unhandled case {}".format(type(data[0])))

            src_keys_and_fn = { "%s_keys" % k : _source_keys_and_feed_fn(k, sa)
                                    for k, sa in feed_sources.items() }

            feed_once_key = key_pool.get(1)
            feed_dict = { ph: getattr(K, n) for n, ph in
                zip(feed_once.fed_arrays, feed_once.placeholders) }
            feed_dict[feed_once.put_key_ph] = feed_once_key[0]
            session.run(feed_once.put_op, feed_dict=feed_dict)

            feed_many_key = key_pool.get(1)
            feed_dict = { ph: getattr(K, n) for n, ph in
                zip(feed_many.fed_arrays, feed_many.placeholders) }
            feed_dict[feed_many.put_key_ph] = feed_many_key[0]
            session.run(feed_many.put_op, feed_dict=feed_dict)

            feed_dict = { ph: src_keys_and_fn[n][0] for n, ph in
                zip(feed_internal.fed_arrays, feed_internal.placeholders) }
            feed_dict[feed_internal.put_key_ph] = feed_many_key[0]
            session.run(feed_internal.put_op, feed_dict=feed_dict)

            # Now feed the source arrays
            for k, fn in src_keys_and_fn.values():
                fn()

            feed_dict = { local_cpu.feed_once_key: feed_once_key[0],
                          local_cpu.feed_many_key: feed_many_key[0] }
            session.run([exprs[0].stage_feed_once,
                        exprs[0].stage_feed_many,
                        exprs[0].stage_source_data,
                        exprs[0].stage_output,
                        exprs[0].stage_cpu_output],
                            feed_dict=feed_dict)

            # Release all keys
            key_pool.release(feed_once_key)
            key_pool.release(feed_many_key)
            for k, fn in src_keys_and_fn.values():
                key_pool.release(k)

            # TODO: This will, in general not be true
            assert key_pool.all_released()



        def _array_dims(array):
            """ Create array dimensions for da.core.top """
            return tuple(d if isinstance(d, str)
                           else "-".join((str(d), array.name, str(i)))
                           for i, d in enumerate(array.shape))

        input_dim_pairs = tuple(v for n, a in D.items()
                                  for v in (a.name,
                                            _array_dims(input_arrays[n])))

        def _flatten_single_sequences(D):
            """ Simplify tuples and lists of length 1 """
            if isinstance(D, list):
                return (_flatten_single_sequences(D[0])
                        if len(D) == 1
                        else [_flatten_single_sequences(v) for v in D])
            # Don't simplify tuples as these can represent keys
            elif isinstance(D, tuple):
                return (_flatten_single_sequences(D[0])
                        if len(D) == 1
                        else tuple(_flatten_single_sequences(v) for v in D))
            elif isinstance(D, collections.Mapping):
                return { k: _flatten_single_sequences(v)
                            for k, v in D.items() }
            else:
                return D

        pprint(input_dim_pairs)

        predict_name = "predict-" + dask.base.tokenize(*D.values())
        predict = da.core.top(_predict,
            predict_name, ("ntime", "nbl", "nchan", "npol"),
            *input_dim_pairs,
            numblocks={a.name: a.numblocks for a in D.values()})

        predict = _flatten_single_sequences(predict)
        get_keys = predict.keys()

        [predict.update(d.dask) for d in D.values()]
        print("Model vis chunks %s" % (D['model_vis'].chunks,))
        pprint({n: len(D[n].dask) for n in feed_many.keys()})

        pprint({n: D[n].chunks for n in fo})
        pprint({n: D[n].chunks for n in fm})

        client.get(predict, get_keys, sync=True)

        D = client.compute(D)

        pprint(D)

        for f in dd.as_completed([D]):
            continue
            D.result()
