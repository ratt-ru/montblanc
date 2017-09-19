import os

import montblanc

import boltons.cacheutils
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
    return da.tile(ap[0], ds.dims['utime'])

def default_antenna2(ds, schema):
    """ Default antenna 2 """
    ap = default_base_ant_pairs(ds.dims['antenna'],
                                ds.attrs['auto_correlations'])
    return da.tile(ap[1], ds.dims['utime'])

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

    return da.concatenate([da.full(tc, ut, chunks=tc) for ut, tc in zip(unique_times, time_chunks)])

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
        "shape": ("row", "chan", "corr"),
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

    # Fill in chunks and real shape
    for array_name, array_schema in sorted_schema:
        array_schema['chunks'] = tuple(10000 if s == 'rows' else dims.get(s,s)
                                            for s in array_schema['shape'])
        array_schema['rshape'] = tuple(dims.get(s, s) for s in array_schema['shape'])


    coords = { k: np.arange(dims[k]) for k in dims.keys() }
    attrs = { 'schema' : schema, 'auto_correlations': False }

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

    return ds

def montblanc_dataset(xms):
    pass

if __name__ == "__main__":
    print default_dataset()
