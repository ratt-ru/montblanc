import collections

import dask
import dask.array as da
from dask.array.core import getter
from dask.base import tokenize
import numpy as np
try:
    import cytoolz as toolz
except ImportError:
    import toolz
import six

from montblanc.impl.rime.tensorflow.dataset import input_schema
from montblanc.impl.rime.tensorflow.tf_session_cache import tf_session_cache

def _setup_tensorflow(cfg_hash, cfg):
    """ Create a tensorflow session """
    class TensorflowSetup(object):
        """ Encapsulates tensorflow session and other objects """
        def __init__(self, cfg):
            import tensorflow as tf
            from montblanc.impl.rime.tensorflow.tf_graph import (
                                _construct_tensorflow_staging_areas,
                                _construct_tensorflow_expression)
            from montblanc.impl.rime.tensorflow.dataset import (
                                input_schema,
                                output_schema)
            from montblanc.impl.rime.tensorflow.key_pool import KeyPool

            devices = ['/cpu:0']

            with tf.Graph().as_default() as graph:
                feed_data = _construct_tensorflow_staging_areas(
                    input_schema(), output_schema(),
                    ('utime', 'row'), devices)

                exprs = [_construct_tensorflow_expression(feed_data,
                                                        cfg, dev, i)
                                    for i, dev in enumerate(devices)]

                init_op = tf.global_variables_initializer()

            self.feed_data = feed_data
            self.init_op = init_op
            self.exprs = exprs
            self.graph = graph
            config = tf.ConfigProto()
            self.session = session = tf.Session("", config=config, graph=graph)
            self.key_pool = KeyPool()
            session.run(init_op)

        def close(self):
            self.session.close()

        def __enter__(self):
            return self

        def __exit__(self, etype, evalue, etraceback):
            self.close()

    return TensorflowSetup(cfg)

class Rime(object):
    def __init__(self, **kwargs):
        try:
            cfg = kwargs.pop('cfg')
        except KeyError:
            self.set_config({})
        else:
            self.set_config(cfg)

    def set_config(self, cfg):
        """
        Sets the configuration for this object.

        Parameters
        ----------
        cfg : string or file or :class:`collections.Mappping`

            1. If a string it will treated as a filename
            2. If a file, config will be loaded from it in YAML format
            3. If a dictionary
        """

        # Treat strings as filenames to be opened
        if isinstance(cfg, six.string_types):
            cfg = open(cfg, 'r')

        # Treat files as containing yaml
        if isinstance(cfg, file):
            from ruamel.yaml import YAML
            yaml = YAML()

            try:
                cfg_ = yaml.load(cfg)
            finally:
                cfg.close()

            # Set config, handling Nones
            cfg = {} if cfg_ is None else cfg_

        # At this point, should have a dict, validate it
        if isinstance(cfg, collections.Mapping):
            from montblanc.configuration import (config_validator,
                                                raise_validator_errors)

            validator = config_validator()
            validator.validate(cfg)
            raise_validator_errors(validator)
            cfg = validator.document
        else:
            raise ValueError("'cfg' is not a dictionary")

        def _freeze(cfg):
            """
            Make `cfg` immutable. `dict` -> `frozenset`
            and `list` to `tuple`
            """
            if isinstance(cfg, collections.Mapping):
                return frozenset({k: _freeze(v) for k, v
                                        in six.iteritems(cfg)}.items())
            elif isinstance(cfg, (tuple, list)):
                return tuple(_freeze(v) for v in cfg)
            else:
                return cfg

        self._cfg = cfg
        self._cfg_hash = hash(_freeze(cfg))

    def __call__(self, mds):
        """
        Create a dask Array representing the
        computation of the
        `Radio Interferometer Measurement Equation` `(RIME)`
        from inputs on the `mds` Dataset object.

        Parameters
        ----------
        mds : :class:`xarray.Dataset`
            Dataset containing RIME inputs.

        Returns
        -------
        :class:`dask.array.Array`
            Dask array of model visibilities.
        """
        in_schema = input_schema()
        # Extract input variables from the dataset
        inputs = { k: v for k, v in mds.data_vars.items()
                                    if k in in_schema.keys() }

        # This needs be have the same ordered as top_args
        # below so that input names are associated with *args
        # in _rime.
        input_names = inputs.keys()

        # Curry _setup_tensorflow with our config for use in _rime
        # We do this because cfg, as a dict, is not hashable and so is
        # consequently unsuitable for passing to `tf_session_cache().open`.
        # However, we do want to create new sessions whenever the
        # configuration hash changes.
        setup_tf = lambda cfg_hash: _setup_tensorflow(cfg_hash, self._cfg)

        def _rime(*args, **kwargs):
            import numpy as np
            """ Compute chunks of the RIME """
            cfg_hash = kwargs.pop('cfg_hash')

            # Associated input names with arguments
            inputs = {k: v for k, v in zip(input_names, args)}

            # Normalise time_index for this chunk
            # TODO(sjperkins) probably OK since time_index is consecutive
            tindex = inputs["time_index"]
            inputs["time_index"] = tindex - tindex.min()

            # Sanity check time indices as these can be
            # a major cause of segmentation faults.
            utime = inputs["antenna_uvw"].shape[0]
            if not np.all(inputs["time_index"] < utime):
                utimes = np.unique(inputs["time_index"])
                raise ValueError("One of the unique indexes '%s' "
                                "in time_index is greater or equal "
                                "to the number of unique times '%s' "
                                "for this particular chunk. "
                                "Unique time and row chunks must agree. "
                                "See :func:`group_row_chunks`."
                                    % (utimes, utime))

            with tf_session_cache().open(setup_tf, cfg_hash) as S:
                session = S.session
                local_cpu = S.feed_data.local_cpu
                feed_internal = local_cpu.feed_internal
                feed_once = local_cpu.feed_once
                feed_many = local_cpu.feed_many
                feed_sources = S.feed_data.local_cpu.sources
                exprs = S.exprs
                key_pool = S.key_pool

                def _source_keys_and_feed_fn(k, sa):
                    """ Returns (keys, feed function) for given source staging area """

                    # arrays in the staging area to feed
                    arrays = { n: (inputs[n], ph) for n, ph
                                        in zip(sa.fed_arrays, sa.placeholders) }
                    # Get the actual arrays
                    data = [t[0] for t in arrays.values()]

                    if not all(type(data[0]) == type(d) for d in data):
                        raise ValueError("Type mismatch in arrays "
                                         "supplied for {}".format(k))

                    # Handle single ndarray case
                    if isinstance(data[0], np.ndarray):
                        #print("Handling numpy arrays for {}".format(k))
                        if data[0].nbytes == 0:
                            #print("{} is zero-length, ignoring".format(k))
                            return [], lambda: None

                        keys = key_pool.get(1)
                        feed_dict = {ph: d for n, (d, ph) in arrays.items()}
                        feed_dict[sa.put_key_ph] = keys[0]
                        from functools import partial
                        fn = partial(session.run, sa.put_op, feed_dict=feed_dict)
                        return keys, fn

                    # Handle multiple ndarrays in a list case
                    elif isinstance(data[0], list):
                        #print("Handling list of size {} for {}".format(len(data[0]), k))
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
                feed_dict = { ph: inputs[n] for n, ph in
                    zip(feed_once.fed_arrays, feed_once.placeholders) }
                feed_dict[feed_once.put_key_ph] = feed_once_key[0]
                session.run(feed_once.put_op, feed_dict=feed_dict)

                feed_many_key = key_pool.get(1)
                feed_dict = { ph: inputs[n] for n, ph in
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
                _,_,_,_,_,vis, X2 = session.run([exprs[0].stage_feed_once,
                            exprs[0].stage_feed_many,
                            exprs[0].stage_source_data,
                            exprs[0].stage_output,
                            exprs[0].stage_cpu_output,
                            exprs[0].model_vis,
                            exprs[0].chi_squared],
                                feed_dict=feed_dict)

                # Release all keys
                key_pool.release(feed_once_key)
                key_pool.release(feed_many_key)
                key_pool.release(toolz.concat(toolz.pluck(0, src_keys_and_fn.values())))

            # Nest the chi-squared to same level as visibilities
            # This is because they'll have the same structure/number of dimensions
            # but not the same shape
            return vis, np.array(X2, ndmin=vis.ndim, copy=False)

        def _mod_dims(dims):
            """
            Convert "utime" dims to "row" dims.
            After chunking, the number of "row" and "utime" blocks
            should be exactly the same for each array, even though
            their sizes will differ. We do this so that :meth:`dask.array.top`
            will match the blocks of these dimensions together
            """
            return tuple("row" if d == "utime" else d for d in dims)

        def _flatten_singletons(D):
            """ Recursively simplify tuples and lists of length 1 """

            # lists and tuples should remain lists and tuples
            if isinstance(D, list):
                return (_flatten_singletons(D[0]) if len(D) == 1
                        else [_flatten_singletons(v) for v in D])
            elif isinstance(D, tuple):
                return (_flatten_singletons(D[0]) if len(D) == 1
                        else tuple(_flatten_singletons(v) for v in D))
            elif isinstance(D, collections.Mapping):
                return { k: _flatten_singletons(v) for k, v in D.items() }
            else:
                return D

        # Use dask names as tokenize inputs
        tokenize_args = [v.data.name if isinstance(v, da.Array) else v for k, v in inputs.items()]
        token = tokenize(*tokenize_args)
        top_name = '-'.join(("rime", token))
        # Create tuple of flattened (name, dim) pairs
        top_args = [v for var in inputs.values()
                      for v in (var.data.name, _mod_dims(var.dims))]
        # Create numblocks dictionary
        top_numblocks = { v.data.name: v.data.numblocks for v in inputs.values() }

        # Create dask dictionary representing application
        # of the _rime function to inputs
        dsk = da.core.top(_rime,            # Function
                        top_name,           # Output name
                        mds.data.dims,      # Output dimensions
                        *top_args,          # Input names and Dimensions
                        numblocks=top_numblocks,
                        cfg_hash=self._cfg_hash)

        # Flatten any length one tuples and lists
        dsk = _flatten_singletons(dsk)

        keys = dsk.keys()

        mv_name = '-'.join(("model-vis", token))
        x2_name = '-'.join(("chi-squared", token))

        mv_dsk = _flatten_singletons({ (mv_name,) + k[1:]: (getter, k, 0) for k in keys })
        x2_dsk = _flatten_singletons({ (x2_name,) + k[1:]: (getter, k, 1) for k in keys })

        # Now add all graph dependencies of associated inputs
        dsk = toolz.merge(dsk, *(v.data.dask for v in inputs.values()))

        # Infer output data types
        if self._cfg['dtype'] == 'float':
            x2_dtype = np.float32
            mv_dtype = np.complex64
        elif self._cfg['dtype'] == 'double':
            x2_dtype = np.float64
            mv_dtype = np.complex128
        else:
            raise ValueError("Invalid dtype")

        # Construct the model visibility array
        mv_array = da.Array(toolz.merge(mv_dsk, dsk), mv_name,
                        chunks=mds.data.data.chunks, dtype=mv_dtype)

        # Each chi squared sums model visibilities to 1 value
        x2_chunks = tuple(tuple(1 for d in tup) for tup in  mds.data.data.chunks)

        # Construct he chi-squared array
        x2_array = da.Array(toolz.merge(x2_dsk, dsk), x2_name,
                        chunks=x2_chunks, dtype=x2_dtype)

        return mv_array, x2_array

import unittest

class TestDaskRime(unittest.TestCase):
    def test_rime(self):
        dask.set_options(get=dask.get)

        from dataset import default_dataset, group_row_chunks

        # Chunk so that multiple threads are employed
        mds = default_dataset()
        chunks = group_row_chunks(mds, mds.dims['row'] // 10)
        mds = mds.chunk(chunks)

        rime = Rime()
        rime.set_config({'polarisation_type': 'linear'})

        model_vis = rime(mds).compute()
        self.assertTrue(model_vis.shape == mds.data.shape)
        self.assertTrue(tf_session_cache().size() == 1)

        # Now modify the configuration and check that
        # two sessions have been created
        rime.set_config({'polarisation_type': 'circular'})
        model_vis = rime(mds).compute()
        self.assertTrue(tf_session_cache().size() == 2)

if __name__ == "__main__":
    unittest.main()