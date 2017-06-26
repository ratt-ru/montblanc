from __future__ import print_function

import collections
from pprint import pprint, pformat
import threading

import attr
import dask
import dask.array as da
import distributed as dd
import numpy as np

import hypercube
import montblanc
import montblanc.util as mbu
from montblanc.impl.rime.tensorflow.dask_tensorflow import start_tensorflow
from montblanc.impl.rime.tensorflow.RimeSolver import (
    _construct_tensorflow_feed_data,
    _construct_tensorflow_expression,
    _partition,
    _setup_hypercube,
    )

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

        sched_info = client.scheduler_info()

        nr_master=1
        nr_worker=len(sched_info["workers"])-1

        # Create a hypercube for setting up our dask arrays
        cube = create_hypercube()
        print(cube)

        # Take all arrays flagged as input
        iter_dims = ['ntime', 'nbl']
        input_arrays = {a.name: a for a in cube.arrays().itervalues()
                        if 'input' in a.tags}

        src_data_sources, feed_many, feed_once = _partition(iter_dims,
                                                        input_arrays.values())

        feed_once = { a.name: a for a in feed_once }
        feed_many = { a.name: a for a in feed_many }

        fo = feed_once.keys()
        fm = feed_many.keys()

        def _create_dask_arrays(cube):
            """ Create dask arrays """
            def _create_dask_array(array):
                size = cube.dim_global_size(*array.shape)
                chunks = tuple(cube.dim_extent_size(*array.shape, single=False))
                A = da.ones(shape=size, chunks=chunks, dtype=array.dtype)
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

        Klass = attr.make_class("Klass", D.keys())

        def _predict(*args, **kwargs):
            import threading

            import tensorflow as tf

            def _setup_tensorflow():
                from attrdict import AttrDict
                from montblanc.impl.rime.tensorflow.RimeSolver import (
                    _construct_tensorflow_feed_data,
                    _construct_tensorflow_expression)

                TensorflowConfig = attr.make_class("TensorflowConfig", ["session", "feed_data"])

                input_arrays["descriptor"] = AttrDict(dtype=np.int32)

                from tensorflow.python.client import device_lib
                devices = device_lib.list_local_devices()

                with tf.Graph().as_default() as compute_graph:
                    shards_per_device = spd = 2
                    shards = len(devices)*spd
                    shard = lambda d, s: d*spd + s

                    # Create our data feeding structure containing
                    # input/output staging_areas and feed once variables
                    feed_data = _construct_tensorflow_feed_data(
                        input_arrays, cube, iter_dims, shards)

                    # Construct tensorflow expressions for each shard
                    exprs = [_construct_tensorflow_expression(
                            feed_data, dev, shard(d,s))
                        for d, dev in enumerate([d.name for d in devices])
                        for s in range(shards_per_device)]

                    # Initialisation operation
                    init_op = tf.global_variables_initializer()
                    # Now forbid modification of the graph
                    compute_graph.finalize()

                session = tf.Session("", graph=compute_graph)
                session.run(init_op)

                return TensorflowConfig(session, feed_data)

            w = dd.get_worker()

            if not hasattr(w, "_thread_local"):
                w._thread_local = tl = threading.local()
                tl.tf_cfg = _setup_tensorflow()
            else:
                tl = w._thread_local

            print(tl.tf_cfg)

            K = Klass(*args)
            D = attr.asdict(K)

            def _display(v):
                if isinstance(v, np.ndarray):
                    return "ndarray{}".format(v.shape,)
                elif isinstance(v, collections.Sequence):
                    return "sequence[{}]".format(len(v))
                else:
                    return v

            pprint({ k: _display(v) for k, v in D.items() })

        def _array_dims(array):
            """ Create array dimensions for da.core.top """
            return tuple(d if isinstance(d, str)
                           else "_".join((str(d), array.name, str(i)))
                           for i, d in enumerate(array.shape))

        def _fix(D):
            """ Simplify lists of length 1 """
            if isinstance(D, list):
                return _fix(D[0]) if len(D) == 1 else [_fix(v) for v in D]
            elif isinstance(D, tuple):
                return _fix(D[0]) if len(D) == 1 else tuple(_fix(v) for v in D)
            elif isinstance(D, collections.Mapping):
                return { k: _fix(v) for k, v in D.items() }
            else:
                return D

        input_dim_pairs = tuple(v for n, a in input_arrays.items()
                                for v in (D[n].name, _array_dims(a)))

        print(input_dim_pairs)

        predict_name = "predict-" + dask.base.tokenize(*D.keys())
        predict = da.core.top(_predict, predict_name,
            ("ntime", "nbl", "nchan", "npol"),
            *input_dim_pairs,
            numblocks={a.name: a.numblocks for a in D.values()})

        predict = _fix(predict)
        get_keys = predict.keys()
        pprint(predict)

        [predict.update(d.dask) for d in D.values()]


        client.get(predict, get_keys, sync=True)



        print("Model vis chunks %s" % (D['model_vis'].chunks,))
        pprint({n: len(D[n].dask) for n in feed_many.keys()})

        pprint({n: D[n].chunks for n in fo})
        pprint({n: D[n].chunks for n in fm})

        D = client.compute(D)

        pprint(D)

        for f in dd.as_completed([D]):
            continue
            D.result()
