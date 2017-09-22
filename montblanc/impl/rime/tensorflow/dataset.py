import itertools
import os
import sys

import montblanc
from xarray_ms import xds_from_ms, xds_from_table


import boltons.cacheutils
import dask
import dask.array as da
import six
import numpy as np
import cppimport
import xarray as xr

dsmod = cppimport.imp('montblanc.ext.dataset_mod')

_lru = boltons.cacheutils.LRU(max_size=16)

@boltons.cacheutils.cachedmethod(_lru)
def default_base_ant_pairs(antenna, auto_correlations=False):
    """ Compute base antenna pairs """
    k = 0 if auto_correlations == True else 1
    return np.triu_indices(antenna, k)

def default_antenna1(ds, schema):
    """ Default antenna 1 """
    ap = default_base_ant_pairs(ds.dims['antenna'],
                                ds.attrs['auto_correlations'])
    return da.from_array(np.tile(ap[0], ds.dims['utime']),
                            chunks=ds.attrs['row_chunks'])

def default_antenna2(ds, schema):
    """ Default antenna 2 """
    ap = default_base_ant_pairs(ds.dims['antenna'],
                                ds.attrs['auto_correlations'])
    return da.from_array(np.tile(ap[1], ds.dims['utime']),
                            chunks=ds.attrs['row_chunks'])

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
    return da.from_array(time, chunks=ds.attrs['row_chunks'])

def default_frequency(ds, schema):
    return da.linspace(8.56e9, 2*8.56e9, schema['rshape'][0],
                                    chunks=schema['chunks'][0])

schema = {
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

def default_schema():
    global schema
    return schema

def default_dataset(**kwargs):
    """
    Creates a default montblanc :class:`xarray.Dataset`

    Returns
    -------
    `xarray.Dataset`
    """
    dims = kwargs.copy()

    # Force these
    dims['(x,y,z)'] = 3
    dims['(u,v,w)'] = 3

    utime = dims.setdefault("utime", 100)
    dims.setdefault("chan", 64)
    dims.setdefault("corr", 4)
    dims.setdefault("pol", 4)
    ants = dims.setdefault("antenna", 7)
    dims.setdefault("spw", 1)

    bl = ants*(ants-1)//2
    dims.setdefault("row", utime*bl)

    # Get and sort the default schema
    schema = default_schema()
    sorted_schema = sorted(schema.items())
    row_chunks = 10000

    # Fill in chunks and real shape
    for array_name, array_schema in sorted_schema:
        array_schema['chunks'] = tuple(row_chunks if s == 'rows' else dims.get(s,s)
                                            for s in array_schema['shape'])
        array_schema['rshape'] = tuple(dims.get(s, s) for s in array_schema['shape'])


    coords = { k: np.arange(dims[k]) for k in dims.keys() }
    attrs = { 'schema' : schema,
                'auto_correlations': False,
                'row_chunks': row_chunks }

    # Create an empty dataset, but with coordinates set
    ds = xr.Dataset(None, coords=coords, attrs=attrs)

    # Create Dataset arrays
    for array_name, array_schema in sorted_schema:
        acoords = { k: coords[k] for k in array_schema['shape']}
        default = lambda ds, as_: da.zeros(shape=array_schema['rshape'],
                                            dtype=as_['dtype'],
                                            chunks=as_['chunks'])
        default = array_schema.get('default', default)

        array = default(ds, array_schema)

        ds[array_name] = xr.DataArray(array, coords=acoords, dims=array_schema['shape'])

    return ds.chunk({"row": 10000})

from pprint import pformat

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

    row_groups = xds.row_groups.values
    utime_groups = xds.utime_groups.values

    token = dask.base.tokenize(xds.uvw, xds.antenna1, xds.antenna2,
                            xds.time_chunks, xds.row_groups, xds.utime_groups)
    name = "-".join(("create_antenna_uvw", token))
    p_ant_uvw = partial(dsmod.antenna_uvw, nr_of_antenna=xds.dims["antenna"])

    def _chunk_iter(chunks):
        start = 0
        for size in chunks:
            end = start + size
            yield slice(start, end)
            start = end

    it = itertools.izip(_chunk_iter(xds.row_groups.values),
                        _chunk_iter(xds.utime_groups.values))

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

def group_rows(xds, max_group_size=100000):
    """
    Adds `row_groups` and `utime_groups` to the :class:`xarray.Dataset`

    Returns
    -------
    `xarray.Dataset`
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

    return xds.assign(row_groups=xr.DataArray(row_groups[1:], dims=["group"]),
                    utime_groups=xr.DataArray(utime_groups[1:], dims=["group"]))


def montblanc_dataset(xds):
    """
    Massages an :class:`xarray.Dataset` produced by `xarray-ms` into
    a dataset expected by montblanc.

    Returns
    -------
    `xarray.Dataset`
    """
    mds = create_time_index(xds)

    # Verify schema
    for k, v in six.iteritems(default_schema()):
        dims = mds[k].dims
        if not dims == v["shape"]:
            raise ValueError("Array '%s' dimensions '%s' does not "
                            "match schema shape '%s'" % (k, dims, v["shape"]))

    return mds

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
    start = np.log2(start)
    size = np.log2(size)
    int_values = np.int32(np.logspace(start, size, div, base=2)[:-1])

    return np.flipud(np.unique(int_values))

def _reduction():
    utimes = _uniq_log2_range(1, mds.dims['utime'], 50)

    for utime in utimes:
        yield [('utime', utime), ('row', mds.time_chunks[:utime].values.sum())]


if __name__ == "__main__":
    from pprint import pprint
    xds = montblanc_dataset(default_dataset())
    print xds
    ms = "~/data/D147-LO-NOIFS-NOPOL-4M5S.MS"

    renames = {'rows': 'row',
                'chans' : 'chan',
                'pols': 'pol',
                'corrs' : 'corr'}

    xds = dataset_from_ms(ms).rename(renames)
    mds = montblanc_dataset(xds)

    ar = budget(mds, 5*1024*1024*1024, _reduction)

    pprint(ar)

    mds = group_rows(mds, max_group_size=ar.get('row'))
    mds = create_antenna_uvw(mds)
    print mds.antenna_uvw

    print mds
    print mds.row_groups.values
    print mds.utime_groups.values
