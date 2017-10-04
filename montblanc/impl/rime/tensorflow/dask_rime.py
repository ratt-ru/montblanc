import collections
from pprint import pprint

import dask
import dask.array as da
try:
    import cytoolz as toolz
except ImportError:
    import toolz
import six

from dataset import input_schema

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

        self._cfg = cfg

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

        def _rime(*args, **kwargs):
            """ Compute chunks of the RIME """

            # TODO(sjperkins): This just passes data straight through
            # Plug tensorflow code in here.
            inputs = {k: v for k, v in zip(input_names, args)}
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
                        numblocks=top_numblocks)

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

        rime = Rime()

        model_vis = rime(mds).compute()
        self.assertTrue(model_vis.shape == mds.data.shape)
        self.assertTrue(da.all(model_vis == mds.data).compute())

if __name__ == "__main__":
    unittest.main()