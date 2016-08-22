import collections
import copy
import threading
import types

import pyrap.tables as pt
import numpy as np
import hypercube as hc
import tensorflow as tf

from montblanc.impl.rime.tensorflow.feeders.feed_context import FeedContext

class RimeDataFeeder(object):
    pass
    
class NumpyRimeDataFeeder(RimeDataFeeder):
    def __init__(self, arrays):
        pass

DOUBLE = 'double'
SINGLE = 'single'

# Map MS column string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'INT' : np.int32,
    'FLOAT' : np.float32,
    'DOUBLE' : np.float64,
    'BOOLEAN' : np.bool,
    'COMPLEX' : np.complex64,
    'DCOMPLEX' : np.complex128
}

SINGLE_TO_DOUBLE_CAST_MAP = {
    'COMPLEX' : 'DCOMPLEX',
    'FLOAT' : 'DOUBLE',
}

DOUBLE_TO_SINGLE_CAST_MAP = { v: k for
    k, v in SINGLE_TO_DOUBLE_CAST_MAP.iteritems() }


# Key names for main and taql selected tables
MAIN_TABLE = 'MAIN'
ORDERED_MAIN_TABLE = 'ORDERED_MAIN'
ORDERED_UVW_TABLE = 'ORDERED_UVW'

# Measurement Set sub-table name string constants
ANTENNA_TABLE = 'ANTENNA'
SPECTRAL_WINDOW_TABLE = 'SPECTRAL_WINDOW'
DATA_DESCRIPTION_TABLE = 'DATA_DESCRIPTION'
POLARIZATION_TABLE = 'POLARIZATION'

SUBTABLE_KEYS = (ANTENNA_TABLE,
    SPECTRAL_WINDOW_TABLE,
    DATA_DESCRIPTION_TABLE,
    POLARIZATION_TABLE)

# String constants for column names
TIME = 'TIME'
ANTENNA1 = 'ANTENNA1'
ANTENNA2 = 'ANTENNA2'
UVW = 'UVW'
DATA = 'DATA'
FLAG = 'FLAG'
WEIGHT = 'WEIGHT'

# Columns used in select statement
SELECTED = [TIME, ANTENNA1, ANTENNA2, UVW, DATA, FLAG, WEIGHT]

# Named tuple defining a mapping from MS row to dimension
OrderbyMap = collections.namedtuple("OrderbyMap", "dimension orderby")

# Mappings for time, baseline and band
TIME_MAP = OrderbyMap("ntime", "TIME")
BASELINE_MAP = OrderbyMap("nbl", "ANTENNA1, ANTENNA2")
BAND_MAP = OrderbyMap("nbands", "[SELECT SPECTRAL_WINDOW_ID "
        "FROM ::DATA_DESCRIPTION][DATA_DESC_ID]")

# Place mapping in a list
MS_ROW_MAPPINGS = [
    TIME_MAP,
    BASELINE_MAP,
    BAND_MAP
]

# Main measurement set ordering dimensions
MS_DIM_ORDER = ('ntime', 'nbl', 'nbands')
# UVW measurement set ordering dimensions
UVW_DIM_ORDER = ('ntime', 'nbl')
DUMMY_CACHE_VALUE = (-1, None)

def orderby_clause(dimensions, unique=False):
    columns = ", ".join(m.orderby for m
        in MS_ROW_MAPPINGS if m.dimension in dimensions)
    
    return " ".join(("ORDERBY", "UNIQUE" if unique else "", columns))

def select_columns(dimensions, dtypes, precision=None):
    """
    Generate select columns. columns will be casted according
    specified precision
    """
    if precision is None:
        precision = DOUBLE

    if precision == DOUBLE:
        dtypes = [SINGLE_TO_DOUBLE_CAST_MAP.get(d, d) for d in dtypes]
    elif precision == SINGLE:
        dtypes = [DOUBLE_TO_SINGLE_CAST_MAP.get(d, d) for d in dtypes]
    else:
        raise ValueError("Invalid precision '{p}'".format(p=precision))

    return ", ".join('{n} AS {n} {d}'.format(n=n, d=d)
        for n, d in zip(dimensions, dtypes))

def subtable_name(msname, subtable=None):
    return '::'.join((msname, subtable)) if subtable else msname

def open_table(msname, subtable=None):
    return pt.table(subtable_name(msname, subtable), ack=False)

class MSRimeDataFeeder(RimeDataFeeder):
    def __init__(self, msname, precision=None):
        super(MSRimeDataFeeder, self).__init__()

        if precision is None:
            precision = DOUBLE

        self._msname = msname
        # Create dictionary of tables
        self._tables = { k: open_table(msname, k) for k in SUBTABLE_KEYS }
        self._cube = cube = hc.HyperCube()

        # Open the main measurement set
        ms = open_table(msname)

        # Access individual tables
        ant, spec, ddesc, pol = (self._tables[k] for k in SUBTABLE_KEYS)

        # Sanity check the polarizations
        if pol.nrows() > 1:
            raise ValueError("Multiple polarization configurations!")

        npol = pol.getcol('NUM_CORR')[0]

        if npol != 4:
            raise ValueError('Expected four polarizations')

        # Number of channels per band
        chan_per_band = spec.getcol('NUM_CHAN')

        # Require the same number of channels per band
        if not all(chan_per_band[0] == cpb for cpb in chan_per_band):
            raise ValueError('Channels per band {cpb} are not equal!'
                .format(cpb=chan_per_band))

        if ddesc.nrows() != spec.nrows():
            raise ValueError("DATA_DESCRIPTOR.nrows() "
                "!= SPECTRAL_WINDOW.nrows()")

        # Hard code auto-correlations and field_id 0
        auto_correlations = True
        field_id = 0

        select_cols = select_columns(SELECTED,
            [ms.getcoldesc(c)["valueType"].upper() for c in SELECTED],
            precision=precision)

        print select_cols

        # Create a view over the MS, ordered by
        # (1) time (TIME)
        # (2) baseline (ANTENNA1, ANTENNA2)
        # (3) band (SPECTRAL_WINDOW_ID via DATA_DESC_ID)
        ordering_query = " ".join((
            "SELECT {c} FROM $ms".format(c=select_cols),
            "WHERE FIELD_ID={fid}".format(fid=field_id),
            "" if auto_correlations else "AND ANTENNA1 != ANTENNA2",
            orderby_clause(MS_DIM_ORDER)
        ))

        # Ordered Measurement Set
        oms = pt.taql(ordering_query)
        # Measurement Set ordered by unique time and baseline
        otblms = pt.taql("SELECT FROM $oms {c}".format(
            c=orderby_clause(UVW_DIM_ORDER, unique=True)))

        # Store the main table
        self._tables[MAIN_TABLE] = ms
        self._tables[ORDERED_MAIN_TABLE] = oms
        self._tables[ORDERED_UVW_TABLE] = otblms

        # Count distinct timesteps in the MS
        t_orderby = orderby_clause(['ntime'], unique=True)
        t_query = "SELECT FROM $otblms {c}".format(c=t_orderby)
        ntime = pt.taql(t_query).nrows()
        
        # Count number of baselines in the MS
        bl_orderby = orderby_clause(['nbl'], unique=True)
        bl_query = "SELECT FROM $otblms {c}".format(c=bl_orderby)
        nbl = pt.taql(bl_query).nrows()

        # Register dimensions on the cube
        cube.register_dimension('npol', npol,
            description='Polarisations')
        cube.register_dimension('nbands', len(chan_per_band),
            description='Bands')
        cube.register_dimension('nchan', sum(chan_per_band),
            description='Channels')
        cube.register_dimension('nchanperband', chan_per_band[0],
            description='Channels-per-band')
        cube.register_dimension('nrows', ms.nrows(),
            description='Main MS rows')
        cube.register_dimension('nuvwrows', otblms.nrows(),
            description='UVW sub-MS rows')
        cube.register_dimension('na', ant.nrows(),
            description='Antenna')
        cube.register_dimension('ntime', ntime,
            description='Timesteps')
        cube.register_dimension('nbl', nbl,
            description='Baselines')

        def _cube_row_update_function(self):
            # Update main measurement set rows
            shape = self.dim_global_size(*MS_DIM_ORDER)
            lower = self.dim_lower_extent(*MS_DIM_ORDER)
            upper = tuple(u-1 for u in self.dim_upper_extent(*MS_DIM_ORDER))

            self.update_dimension(name='nrows',
                lower_extent=np.ravel_multi_index(lower, shape),
                upper_extent=np.ravel_multi_index(upper, shape)+1)

            shape = self.dim_global_size(*UVW_DIM_ORDER)
            lower = self.dim_lower_extent(*UVW_DIM_ORDER)
            upper = tuple(u-1 for u in self.dim_upper_extent(*UVW_DIM_ORDER))

            self.update_dimension(name='nuvwrows',
                lower_extent=np.ravel_multi_index(lower, shape),
                upper_extent=np.ravel_multi_index(upper, shape)+1)

        self._cube.update_row_dimensions = types.MethodType(
            _cube_row_update_function, self._cube)

        # Temporary, need to get these arrays from elsewhere
        cube.register_array('uvw', ('ntime', 'na', 3), np.float64)
        cube.register_array('antenna1', ('ntime', 'nbl'), np.int32)
        cube.register_array('antenna2', ('ntime', 'nbl'), np.int32)
        cube.register_array('observed_vis', ('ntime', 'nbl', 'nchan', 'npol'), np.complex64)
        cube.register_array('weight', ('ntime', 'nbl', 'nchan', 'npol'), np.float32)
        cube.register_array('flag', ('ntime', 'nbl', 'nchan', 'npol'), np.bool)

    @property
    def mscube(self):
        return self._cube

    def _hash_array_idx(self, name, cube):
        D = cube.dimensions(copy=False)
        idx = ((D[d].lower_extent, D[d].upper_extent) if d in D 
            else (0, d) for d in cube.array(name).shape)
        return hash(i for i in ((s[0], s[1]) for s in idx))
    
    def uvw(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        ntime, nbl, na = context.dim_extent_size('ntime', 'nbl', 'na')

        bl_uvw = self._tables[ORDERED_UVW_TABLE].getcol(UVW,
            startrow=lrow, nrow=urow-lrow).reshape(ntime, nbl, 3)

        ant_uvw = np.empty(shape=(ntime, na, 3),dtype=bl_uvw.dtype)
        ant_uvw[:,1:na,:] = bl_uvw[:,:na-1,:]
        ant_uvw[:,0,:] = 0

        return ant_uvw

    def antenna1(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        antenna1 = self._tables[ORDERED_MAIN_TABLE].getcol(
            ANTENNA1, startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape)

    def antenna2(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        antenna2 = self._tables[ORDERED_MAIN_TABLE].getcol(
            ANTENNA2, startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape)

    def observed_vis(self, context):
        lrow, urow = context.dim_extents('nuvwrows')

        data = self._tables[ORDERED_MAIN_TABLE].getcol(
            DATA, startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape)

    def flag(self, context):
        lrow, urow = context.dim_extents('nuvwrows')

        flag = self._tables[ORDERED_MAIN_TABLE].getcol(
            FLAG, startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape)

    def weight(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        nchan = context.dim_extent_size('nchanperband')

        weight = self._tables[ORDERED_MAIN_TABLE].getcol(
            WEIGHT, startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        weight = np.repeat(weight, nchan, 0)
        return weight.reshape(context.shape)

    def close(self):
        for table in self._tables.itervalues():
            table.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('msfile')
args = parser.parse_args()

feeder = MSRimeDataFeeder(args.msfile, precision=SINGLE)
cube = copy.deepcopy(feeder.mscube)

row_iter_sizes = [10] + cube.dim_global_size('nbl', 'nbands')
dim_iter_args = zip(MS_DIM_ORDER, row_iter_sizes)

# Arrays that we will feed
array_names = ('antenna1', 'antenna2', 'uvw',
        'observed_vis', 'flag', 'weight')

for dims in cube.dim_iter(*dim_iter_args, update_local_size=True):
    cube.update_dimensions(dims)
    cube.update_row_dimensions()
    arrays = cube.arrays(reify=True)

    feed_contexts = ((n, FeedContext(n,
        cube, {}, arrays[n].shape, arrays[n].dtype))
        for n in array_names)

    feed_arrays = ((n, getattr(feeder, n)(c)) for n, c in feed_contexts)

    print ' '.join(['{n} {s}'.format(n=n,s=a.shape) for n, a in feed_arrays])

