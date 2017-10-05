import collections
from pprint import pprint

import dask
import dask.array as da
try:
    import cytoolz as toolz
except ImportError:
    import toolz
import six

from montblanc.impl.rime.tensorflow.dataset import input_schema, output_schema
from montblanc.impl.rime.tensorflow.tf_session_cache import tf_session_cache

def _create_tf_session(cfg_hash, cfg):
    """ Create a tensorflow session """
    import tensorflow as tf
    from tf_graph import (_construct_tensorflow_staging_areas,
                        _construct_tensorflow_expression)

    devices = ['/cpu:0']

    with tf.Graph().as_default() as graph:
        feed_data = _construct_tensorflow_staging_areas(
            input_schema(),
            output_schema(),
            ('utime', 'row'),
            devices)

        expr = _construct_tensorflow_expression(feed_data,
                                                cfg,
                                                devices[0],
                                                0)

        init_op = tf.global_variables_initializer()

    session = tf.Session("", graph=graph)
    session.run(init_op)
    #return graph, init_op, expr, feed_data

    return session


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

        in_schema = input_schema()
        # Extract input variables from the dataset
        inputs = { k: v for k, v in mds.data_vars.items()
                                    if k in in_schema.keys() }

        # This needs be have the same ordered as top_args
        # below so that input names are associated with *args
        # in _rime.
        input_names = inputs.keys()

        # Curry _create_tf_session with our config for use in _rime
        # We do this because cfg, as a dict, is not hashable and so is
        # consequently unsuitable for passing to `tf_session_cache().open`.
        # However, we do want to create new sessions whenever the
        # configuration hash changes.
        mk_tf_sess = lambda cfg_hash: _create_tf_session(cfg_hash, self._cfg)

        def _rime(*args, **kwargs):
            """ Compute chunks of the RIME """
            cfg_hash = kwargs.pop('cfg_hash')

            # TODO(sjperkins): This just passes data straight through
            # Plug tensorflow code in here.
            inputs = {k: v for k, v in zip(input_names, args)}

            with tf_session_cache().open(mk_tf_sess, cfg_hash) as S:
                pass

            return inputs['data']

        # Use dask names ask tokenize inputs
        tokenize_args = [v.data.name for k, v in inputs.items()]
        top_name = '-'.join(("rime", dask.base.tokenize(*tokenize_args)))
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

        # Flatten tuples/list of length 1 and
        # add dask graphs of associated inputs
        dsk = toolz.merge(_flatten_singletons(dsk),
                        *(v.data.dask for v in inputs.values()))


        return da.Array(dsk, top_name,
                        chunks=mds.data.data.chunks,
                        dtype=mds.data.dtype)

import unittest

class TestDaskRime(unittest.TestCase):
    def test_rime(self):
        from dataset import default_dataset

        mds = default_dataset()

        # Chunk so that multiple threads are employed
        dims = mds.dims
        rows_per_utime = dims['row'] // dims['utime']
        utime = dims['utime'] // 10
        row = utime*rows_per_utime

        mds = mds.chunk({'utime':utime, 'row': row})

        rime = Rime()
        rime.set_config({'polarisation_type': 'linear'})

        model_vis = rime(mds).compute()
        self.assertTrue(model_vis.shape == mds.data.shape)
        self.assertTrue(da.all(model_vis == mds.data).compute())
        self.assertTrue(tf_session_cache().size() == 1)

        # Now modify the configuraiton and check that
        # two sessions have been created
        rime.set_config({'polarisation_type': 'circular'})
        model_vis = rime(mds).compute()
        self.assertTrue(tf_session_cache().size() == 2)

if __name__ == "__main__":
    unittest.main()