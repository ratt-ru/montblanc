import collections
import itertools
import os
import sys

import boltons.cacheutils
import cppimport
import dask
import dask.array as da
import numpy as np
import six
try:
    import cytoolz as toolz
except ImportError:
    import toolz
import xarray as xr
from xarray_ms import xds_from_ms, xds_from_table

import montblanc

dsmod = cppimport.imp('montblanc.ext.dataset_mod')

def default_base_ant_pairs(antenna, auto_correlations=False):
    """ Compute base antenna pairs """
    k = 0 if auto_correlations == True else 1
    return np.triu_indices(antenna, k)

def default_antenna1(ds, schema):
    """ Default antenna 1 """
    ap = default_base_ant_pairs(ds.dims['antenna'],
                                ds.attrs['auto_correlations'])
    return da.from_array(np.tile(ap[0], ds.dims['utime']),
                            chunks=ds.chunks['row'])

def default_antenna2(ds, schema):
    """ Default antenna 2 """
    ap = default_base_ant_pairs(ds.dims['antenna'],
                                ds.attrs['auto_correlations'])
    return da.from_array(np.tile(ap[1], ds.dims['utime']),
                            chunks=ds.chunks['row'])

def default_time_unique(ds, schema):
    """ Default unique time """
    return np.linspace(4.865965e+09, 4.865985e+09,
                        schema['rshape'][0])

def default_time_offset(ds, schema):
    """ Default time offset """
    row, utime = (ds.dims[k] for k in ('row', 'utime'))

    bl = row // utime
    assert utime*bl == row
    return np.arange(utime)*bl

def default_time_chunks(ds, schema):
    """ Default time chunks """
    row, utime = (ds.dims[k] for k in ('row', 'utime'))

    bl = row // utime
    assert utime*bl == row
    return np.full(schema['rshape'], bl)

def default_time(ds, schema):
    """ Default time """
    unique_times = default_time_unique(ds, ds.attrs['schema']['time_unique'])
    time_chunks = default_time_chunks(ds, ds.attrs['schema']['time_chunks'])

    time = np.concatenate([np.full(tc, ut) for ut, tc in zip(unique_times, time_chunks)])
    return da.from_array(time, chunks=ds.chunks['row'])

def default_frequency(ds, schema):
    return da.linspace(8.56e9, 2*8.56e9, schema['rshape'][0],
                                    chunks=schema['chunks'][0])

def scratch_schema():
    return {
        "bsqrt": {
            "shape": ("source", "utime", "chan", "corr"),
            "dtype": np.complex128,
        },

        "complex_phase": {
            "shape": ("source", "utime", "antenna", "chan"),
            "dtype": np.complex128,
        },

        "ejones": {
            "shape": ("source", "utime", "antenna", "chan", "corr"),
            "dtype": np.complex128,
        },

        "antenna_jones": {
            "shape": ("source", "utime", "antenna", "chan", "corr"),
            "dtype": np.complex128,
        },

        "sgn_brightness": {
            "shape": ("source", "utime"),
            "dtype": np.int8,
        },

        "source_shape": {
            "shape": ("source", "row", "chan"),
            "dtype": np.float64,
        },

        "chi_sqrd_terms": {
            "shape": ("row", "chan"),
            "dtype": np.float64,
        }
    }

def source_schema():
    return {
        "point_lm": {
            "shape": ("point", "(l,m)"),
            "dtype": np.float64,
        },
        "point_stokes": {
            "shape": ("point", "utime", "(I,Q,U,V)"),
            "dtype": np.float64,
        },
        "gaussian_lm": {
            "shape": ("gaussian", "(l,m)"),
            "dtype": np.float64,
        },
        "gaussian_stokes": {
            "shape": ("gaussian", "utime", "(I,Q,U,V)"),
            "dtype": np.float64,
        },
        "gaussian_shape_params": {
            "shape": ("gaussian", "(lproj,mproj,theta)"),
            "dtype": np.float64,
        },
        "sersic_lm": {
            "shape": ("sersic", "(l,m)"),
            "dtype": np.float64,
        },
        "sersic_stokes": {
            "shape": ("sersic", "utime", "(I,Q,U,V)"),
            "dtype": np.float64,
        },
        "sersic_shape_params": {
            "shape": ("sersic", "(s1,s2,theta)"),
            "dtype": np.float64,
        },

    }

def default_schema():
    return {
        "time" : {
            "shape": ("row",),
            "dtype": np.float64,
            "default": default_time,
        },

        "time_unique": {
            "shape": ("utime",),
            "dtype": np.float64,
            "default": default_time_unique,
        },

        "time_offsets" : {
            "shape": ("utime",),
            "dtype": np.int32,
            "default": default_time_offset,
        },

        "time_chunks" : {
            "shape": ("utime",),
            "dtype": np.int32,
            "default": default_time_chunks,
        },

        "model_data": {
            "shape": ("row", "chan", "corr"),
            "dtype": np.complex128,
        },

        "uvw": {
            "shape": ("row", "(u,v,w)"),
            "dtype": np.float64,
        },

        "antenna1" : {
            "shape": ("row",),
            "dtype": np.int32,
            "default": default_antenna1,
        },

        "antenna2" : {
            "shape": ("row",),
            "dtype": np.int32,
            "default": default_antenna2,
        },

        "flag": {
            "shape": ("row", "chan", "corr"),
            "dtype": np.bool,
            "default": lambda ds, as_: da.full(as_["rshape"], False,
                                                dtype=as_["dtype"],
                                                chunks=as_["chunks"])
        },

        "weight": {
            "shape": ("row", "corr"),
            "dtype": np.float32,
            "default": lambda ds, as_: da.ones(shape=as_["rshape"],
                                                dtype=as_["dtype"],
                                                chunks=as_["chunks"])
        },

        "frequency": {
            "shape": ("chan",),
            "dtype": np.float64,
            "default": default_frequency,
        },

        "antenna_position": {
            "shape": ("antenna", "(x,y,z)"),
            "dtype": np.float64,
        },
    }

def default_dim_sizes():
    """ Returns a dictionary of default dimension sizes """
    ds = {
        '(I,Q,U,V)': 4,
        '(x,y,z)': 3,
        '(u,v,w)': 3,
        'utime': 100,
        'chan': 64,
        'corr': 4,
        'pol': 4,
        'antenna': 7,
        'spw': 1,
    }

    nbl = ds['antenna']*(ds['antenna']-1)//2
    ds.update({'row': ds['utime']*nbl })

    ds.update({
        'point': 1,
        'gaussian': 1,
        'sersic': 1,
        '(l,m)': 2,
        '(lproj,mproj,theta)': 3,
        '(s1,s2,theta)': 3,
    })

    return ds


def input_schema():
    """ Montblanc input schemas """
    return toolz.merge(default_schema(), source_schema())

def default_dataset(xds=None):
    """
    Creates a default montblanc :class:`xarray.Dataset`.(
        If `xds` is supplied, missing arrays will be filled in
        with default values.

        Parameters
        ----------
        xds (optional): :class:`xarray.Dataset`

    Returns
    -------
    :class:`xarray.Dataset`
    """

    dims = default_dim_sizes()
    in_schema = toolz.merge(default_schema(), source_schema())

    if xds is None:
        # Create coordinates for each dimension
        coords = { k: np.arange(dims[k]) for k in dims.keys() }
        # Create a dummy array with shape ('row',) so that there is
        # a chunking strategy along this dimension. Needed for most default
        # methods
        arrays = { "__dummy__" : xr.DataArray(da.ones(shape=dims['row'],
                                                        chunks=10000,
                                                        dtype=np.float64),
                                                dims=["row"]) }
        xds = xr.Dataset(arrays, coords=coords)
    else:
        # Create coordinates for default dimensions
        # not present on the dataset
        coords = { k: np.arange(dims[k]) for k in dims.keys()
                                        if k not in xds.dims }

        # Update dimension dictionary with dataset dimensions
        dims.update(xds.dims)

        # Assign coordinates
        xds.assign_coords(**coords)

    default_attrs = { 'schema': in_schema,
                       'auto_correlations': False }

    default_attrs.update(xds.attrs)
    xds.attrs.update(default_attrs)

    arrays = xds.data_vars.keys()
    missing_arrays = set(in_schema).difference(arrays)

    chunks = xds.chunks

    # Create reified shape and chunks on missing array schemas
    for n in missing_arrays:
        schema = in_schema[n]
        sshape = schema["shape"]
        schema["rshape"] = rshape = tuple(dims.get(d, d) for d in sshape)
        schema["chunks"] = tuple(chunks.get(d, r) for d, r in zip(sshape, rshape))

    def _default_zeros(ds, schema):
        """ Return a dask array of zeroes """
        return da.zeros(shape=schema['rshape'],
                       chunks=schema['chunks'],
                        dtype=schema['dtype'])

    def _create_array(array):
        """ Create array """
        schema = in_schema[array]
        default = schema.get('default', _default_zeros)
        return xr.DataArray(default(xds, schema), dims=schema['shape'])

    missing_arrays = { n: _create_array(n) for n in missing_arrays }

    xds = xds.assign(**missing_arrays)

    # Drop dummy array if present
    if "__dummy__" in xds:
        xds = xds.drop("__dummy__")

    return xds

def create_antenna_uvw(xds):
    """
    Adds `antenna_uvw` coordinates to the given :class:`xarray.Dataset`.

    Returns
    -------
    :class:`xarray.Dataset`
        `xds` with `antenna_uvw` assigned.
    """
    from operator import getitem
    from functools import partial

    row_groups = xds.chunks['row']
    utime_groups = xds.chunks['utime']

    token = dask.base.tokenize(xds.uvw, xds.antenna1, xds.antenna2,
                            xds.time_chunks, row_groups, utime_groups)
    name = "-".join(("create_antenna_uvw", token))
    p_ant_uvw = partial(dsmod.antenna_uvw, nr_of_antenna=xds.dims["antenna"])

    def _chunk_iter(chunks):
        start = 0
        for size in chunks:
            end = start + size
            yield slice(start, end)
            start = end

    it = itertools.izip(_chunk_iter(row_groups),
                        _chunk_iter(utime_groups))

    dsk = { (name, i, 0, 0): (p_ant_uvw,
                                (getitem, xds.uvw, rs),
                                (getitem, xds.antenna1, rs),
                                (getitem, xds.antenna2, rs),
                                (getitem, xds.time_chunks, uts))

                for i, (rs, uts) in enumerate(it) }

    chunks = (tuple(utime_groups), (xds.dims["antenna"],), (xds.dims["(u,v,w)"],))
    dask_array = da.Array(dsk, name, chunks, xds.uvw.dtype)
    dims = ("utime", "antenna", "(u,v,w)")
    return xds.assign(antenna_uvw=xr.DataArray(dask_array, dims=dims))

def create_time_index(xds):
    """
    Adds the `time_index` array specifying the unique time index
    associated with row to the given :class:`xarray.Dataset`.


    Returns
    -------
    :class:`xarray.Dataset`
        `xds` with `time_index` assigned.
    """
    time_chunks = xds.time_chunks.values
    tindices = np.empty(time_chunks.sum(), np.int32)
    start = 0

    for i, c in enumerate(time_chunks):
        tindices[start:start+c] = i
        start += c

    return xds.assign(time_index=xr.DataArray(tindices, dims=('row',)))

def dataset_from_ms(ms):
    """
    Creates an xarray dataset from the given Measurement Set

    Returns
    -------
    `xarray.Dataset`
        Dataset with MS columns as arrays
    """
    xds = xds_from_ms(ms)
    xads = xds_from_table("::".join((ms, "ANTENNA")), table_schema="ANTENNA")
    xspwds = xds_from_table("::".join((ms, "SPECTRAL_WINDOW")), table_schema="SPECTRAL_WINDOW")
    xds = xds.assign(antenna_position=xads.rename({"rows" : "antenna"}).drop('msrows').position,
                    frequency=xspwds.rename({"rows":"spw", "chans" : "chan"}).drop('msrows').chan_freq[0])
    return xds

def merge_dataset(iterable):
    """
    Merge datasets. Dataset dimensions and coordinates must match.
    Later datasets have precedence.

    Parameters
    ----------
    iterable : :class:`xarray.Dataset` or iterable of :class:`xarray.Dataset`
        Datasets to merge

    Returns
    -------
    :class:`xarray.Dataset`
        Merged dataset

    """
    if not isinstance(iterable, collections.Sequence):
        iterable = [iterable]

    # Construct lists of sizes and coordinates for each dimension
    dims = collections.defaultdict(list)
    coords = collections.defaultdict(list)

    for i, ds in enumerate(iterable):
        for dim, size in ds.dims.iteritems():
            # Record dataset index
            dims[dim].append(DimensionInfo(i, size))

        for dim, coord in ds.coords.iteritems():
            coords[dim].append(DimensionInfo(i, coord.values))

    # Sanity check dimension matches on all datasets
    for name, dim_sizes in dims.iteritems():
        if not all(dim_sizes[0].info == ds.info for ds in dim_sizes[1:]):
            msg_str = ','.join(['(dataset=%d,%s=%d)' % (ds.index, name, ds.info)
                                                            for ds in dim_sizes])

            raise ValueError("Conflicting dataset dimension sizes for "
                            "dimension '{n}'. '{ds}'".format(n=name, ds=msg_str))

    # Sanity check dimension coordinates matches on all datasets
    for name, coord in coords.iteritems():
        compare = [(coord[0].info == co.info).all() for co in coord]
        if not all(compare):
            msg_str = ','.join(["(dataset %d '%s' coords match 0: %s)" % (co.index, name, c)
                                            for co, c in zip(dim_sizes, compare)])

            raise ValueError("Conflicting dataset coordinates for "
                            "dimension '{n}'. {m}".format(n=name, m=msg_str))

    # Create dict of data variables for merged datsets
    # Last dataset has precedence
    data_vars = { k: v for ds in iterable
                    for k, v in ds.data_vars.items() }

    # Merge attributes
    attrs = toolz.merge(ds.attrs for ds in iterable)

    return xr.Dataset(data_vars, attrs=attrs)


def group_row_chunks(xds, max_group_size=100000):
    """
    Return a dictionary of unique time and row groups.
    Groups are formed by accumulating chunks in the
    `time_chunks` array attached to `xds` until `max_group_size`
    is reached.

    Parameters
    ----------
    xds : :class:`xarray.Dataset`
        Dataset with `time_chunks` member
    max_group_size (optional) : integer
        Maximum group size

    Returns
    -------
    dict
        { 'utime': (time_group_1, ..., time_group_n),
          'row': (row_group_1, ..., row_group_n) }
    """
    row_groups = [0]
    utime_groups = [0]
    rows = 0
    utimes = 0

    for chunk in xds.time_chunks.values:
        next_ = rows + chunk

        if next_ > max_group_size:
            row_groups.append(rows)
            utime_groups.append(utimes)
            rows = chunk
            utimes = 1
        else:
            rows += chunk
            utimes += 1

    if rows > 0:
        row_groups.append(rows)
        utime_groups.append(utimes)

    return { 'utime': tuple(utime_groups[1:]), 'row': tuple(row_groups[1:]) }

def montblanc_dataset(xds=None):
    """
    Massages an :class:`xarray.Dataset` produced by `xarray-ms` into
    a dataset expected by montblanc.

    Returns
    -------
    `xarray.Dataset`
    """
    if xds is None:
        return default_dataset().drop("uvw")

    schema = input_schema()
    required_arrays = set(schema.keys())
    mds = xds.drop(set(xds.data_vars.keys()).difference(required_arrays))
    mds = default_dataset(mds)
    mds = create_antenna_uvw(mds)

    return mds.drop("uvw")

def budget(xds, mem_budget, reduce_fn):
    """
    Reduce `xds` dimensions using reductions
    obtained from generator `reduce_fn` until
    :code:`xds.nbytes <= mem_budget`.

    Parameters
    ----------
    xds : :class:`array.Dataset`
        xarray dataset
    mem_budget : int
        Number of bytes defining the memory budget
    reduce_fn : callable
        Generator yielding a lists of dimension reduction tuples.
        For example:

        .. code-block:: python

            def red_gen():
                yield [('utime', 100), ('row', 10000)]
                yield [('utime', 50), ('row', 1000)]
                yield [('utime', 20), ('row', 100)]

    Returns
    -------
    dict
        A {dim: size} mapping of dimension reductions that
        fit the sliced dataset into the memory budget.
    """
    bytes_required = xds.nbytes
    applied_reductions = {}
    mds = xds

    for reduction in reduce_fn():
        if bytes_required > mem_budget:
            mds = mds.isel(**{ dim: slice(0, size) for dim, size in reduction })
            applied_reductions.update({ dim: size for dim, size in reduction })
            bytes_required = mds.nbytes
        else:
            break

    return applied_reductions

def _uniq_log2_range(start, size, div):
    """
    Produce unique integers in the start, start+size range
    with a log2 distribution
    """
    start = np.log2(start)
    size = np.log2(size)
    int_values = np.int32(np.logspace(start, size, div, base=2)[:-1])

    return np.flipud(np.unique(int_values))

def _reduction():
    """ Default reduction """
    utimes = _uniq_log2_range(1, mds.dims['utime'], 50)

    for utime in utimes:
        rows = mds.time_chunks[:utime].values.sum()
        yield [('utime', utime), ('row', rows)]

if __name__ == "__main__":
    from pprint import pprint
    xds = montblanc_dataset()
    print xds

    ms = "~/data/D147-LO-NOIFS-NOPOL-4M5S.MS"

    renames = { 'rows': 'row',
                'chans': 'chan',
                'pols': 'pol',
                'corrs': 'corr'}

    xds = dataset_from_ms(ms).rename(renames)
    mds = create_time_index(xds)
    print mds.dims['utime']
    print mds

    ar = budget(mds, 5*1024*1024*1024, _reduction)
    pprint(ar)
    chunks = group_row_chunks(mds, max_group_size=ar['row'])
    mds = mds.chunk(chunks)
    mds = montblanc_dataset(mds)

    # Test antenna_uvw are properly computed. Do not delete!
    print mds.antenna_uvw.compute()

    pprint(dict(mds.chunks))
    pprint(mds.antenna_uvw.chunks)

    def _mod_dims(dims):
        """
        Convert "utime" dims to "row" dims.
        After chunking, the number of "row" and "utime" blocks
        should be exactly the same for each array, even though
        their sizes will differ. We do this so that :meth:`dask.array.top`
        will match the blocks of these dimensions together
        """
        return tuple("row" if d == "utime" else d for d in dims)

    name_dims = [v for var in mds.data_vars.values()
                    for v in (var.data.name, _mod_dims(var.dims))]
    names = [var.data.name for var in mds.data_vars.values()]
    arg_names = [var.name for var in mds.data_vars.values()]
    numblocks = {var.data.name: var.data.numblocks for var in mds.data_vars.values()}

    def _plort(*args):
        """ Predict function. Just pass through `model_data` for now """
        kw = {n: a for n, a in zip(arg_names, args)}

        def _argshape(arg):
            """ Get shapes depending on type """
            if isinstance(arg, np.ndarray):
                return arg.shape
            elif isinstance(args, list):
                return [v.shape for v in arg]
            elif isinstance(args, tuple):
                return tuple(v.shape for v in arg)
            else:
                raise ValueError("Can't infer shape for type '%s'" % type(arg))

        shapes = {n: _argshape(a) for n, a in kw.items()}

        pprint(shapes)
        return kw['model_data']

    # Create a name for this function, constructed from lesser names
    dsk_name = '-'.join(("plort9000", dask.base.tokenize(*names)))
    dsk = da.core.top(_plort, dsk_name, mds.model_data.dims,
                            *name_dims, numblocks=numblocks)

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

    dsk = _flatten_singletons(dsk)

    for n in mds.data_vars.keys():
        dsk.update(getattr(mds, n).data.dask)

    A = da.Array(dsk, dsk_name, chunks=mds.model_data.data.chunks, dtype=mds.model_data.dtype)

    print A
    print A.compute().shape