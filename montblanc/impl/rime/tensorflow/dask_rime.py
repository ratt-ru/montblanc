import collections
from pprint import pprint

import dask
import dask.array as da
try:
    import cytoolz as toolz
except ImportError:
    import toolz

from dataset import input_schema

def rime(mds):
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

        model_vis = rime(mds).compute()
        self.assertTrue(model_vis.shape == mds.data.shape)
        self.assertTrue(da.all(model_vis == mds.data).compute())

if __name__ == "__main__":
    unittest.main()