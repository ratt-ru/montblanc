import collections
import functools
import sys
import time
import types

import pyrap.tables as pt
import numpy as np

import hypercube as hc

import montblanc
from montblanc.impl.rime.tensorflow.feeders.feed_context import FeedContext

class RimeDataFeeder(object):
    pass
    
class NumpyRimeDataFeeder(RimeDataFeeder):
    """
    Given a dictionary containing numpy arrays and keyed on array name,
    provides feed functions for each array.


    >>> feeder = NumpyRimeDataFeeder({
        "uvw" : np.zeros(shape=(100,14,3),dtype=np.float64),
        "antenna1" : np.zeros(shape=(100,351), dtype=np.int32)})

    >>> context = FeedContext(...)
    >>> feeder.uvw(context)
    >>> feeder.antenna1(context)

    """
    def __init__(self, arrays, cube):
        self._arrays = arrays

        def _create_feed_function(name, array):
            def _feed(self, context):
                """ Generic feed function """
                idx = context.slice_index(*context.array(name).shape)
                return array[idx]

            return _feed

        # Create feed methods for each supplied array
        for n, a in arrays.iteritems():
            try:
                array_schema = cube.array(n)
            except KeyError as e:
                # Ignore the array if it isn't defined on the cube
                raise ValueError("Feed array '{n}' is not defined "
                    "on the hypercube.".format(n=n)), None, sys.exc_info()[2]

            # Except the shape of the supplied array to be equal to
            # the size of the global dimensions
            shape = tuple(cube.dim_global_size(d) if isinstance(d, str)
                else d for d in array_schema.shape)

            if shape != a.shape:
                raise ValueError("Shape of supplied array '{n}' "
                    "does not match the global shape '{g}' "
                    "of the array schema '{s}'.".format(
                        n=n, g=shape, s=array_schema.shape))

            # Create the feed function, update the wrapper,
            # bind it to a method and set the attribute on the object
            f = functools.update_wrapper(
                _create_feed_function(n, a),
                _create_feed_function)

            f.__doc__ = "Feed function for array '{n}'".format(n=n)

            method = types.MethodType(f, self)
            setattr(self, n, method)
            

    @property
    def arrays(self):
        return self._arrays

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

def cache_ms_read(method):
    """
    Decorator for caching MSRimeDataFeeder feeder function return values

    Create a key index for the proxied array in the FeedContext.
    Iterate over the array shape descriptor e.g. (ntime, nbl, 3)
    returning tuples containing the lower and upper extents
    of string dimensions. Takes (0, d) in the case of an integer
    dimensions.
    """

    @functools.wraps(method)
    def memoizer(self, context):
        D = context.dimensions(copy=False)
        # (lower, upper) else (0, d)
        idx = ((D[d].lower_extent, D[d].upper_extent) if d in D 
            else (0, d) for d in context.array(context.name).shape)
        # Construct the key for the above index
        key = tuple(i for t in idx for i in t)
        # Access the sub-cache for this array
        array_cache = self._cache[context.name]

        # Cache miss, call the function
        if key not in array_cache:
            array_cache[key] = method(self, context)

        return array_cache[key]

    return memoizer

def orderby_clause(dimensions, unique=False):
    columns = ", ".join(m.orderby for m
        in MS_ROW_MAPPINGS if m.dimension in dimensions)
    
    return " ".join(("ORDERBY", "UNIQUE" if unique else "", columns))

def select_columns(dimensions, dtypes, precision=None):
    """
    Generate select columns. columns will be casted according
    specified precision
    """
    if precision is None or precision == DOUBLE:
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

        self._cache = collections.defaultdict(dict)

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
    
    @cache_ms_read
    def uvw(self, context):
        """ Special case for handling antenna uvw code """

        # Antenna reading code expects (ntime, nbl) ordering
        if UVW_DIM_ORDER != ('ntime', 'nbl'):
            raise ValueError("'{o}'' ordering expected for "
                "antenna reading code.".format(o=UVW_DIM_ORDER))

        (t_low, t_high) = context.dim_extents('ntime')
        na = context.dim_global_size('na')

        # We expect to handle all antenna at once
        if context.shape != (t_high - t_low, na, 3):
            raise ValueError("Received an unexpected shape "
                "{s} in (ntime,na,3) antenna reading code".format(
                    s=context.shape))

        # Create per antenna UVW coordinates.
        # u_01 = u_1 - u_0
        # u_02 = u_2 - u_0
        # ...
        # u_0N = u_N - U_0
        # where N = na - 1.

        # Choosing u_0 = 0 we have:
        # u_1 = u_01
        # u_2 = u_02
        # ...
        # u_N = u_0N

        # Then, other baseline values can be derived as
        # u_21 = u_1 - u_2

        # Allocate space for per-antenna UVW
        ant_uvw = np.empty(shape=context.shape, dtype=context.dtype)
        # Zero antenna 0
        ant_uvw[:,0,:] = 0

        # Read in uvw[1:na] row at each timestep
        for ti, t in enumerate(xrange(t_low, t_high)):
            # Inspection confirms that this achieves the# same effect as 
            # ant_uvw[ti,1:na,:] = ...getcol(UVW, ...).reshape(na-1, -1)
            self._tables[ORDERED_UVW_TABLE].getcolnp(UVW,
                ant_uvw[ti,1:na,:],
                startrow=t*na+1, nrow=na-1)

        return ant_uvw

    @cache_ms_read
    def antenna1(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        antenna1 = self._tables[ORDERED_UVW_TABLE].getcol(
            ANTENNA1, startrow=lrow, nrow=urow-lrow)

        return antenna1.reshape(context.shape)

    @cache_ms_read
    def antenna2(self, context):
        lrow, urow = context.dim_extents('nuvwrows')
        antenna2 = self._tables[ORDERED_UVW_TABLE].getcol(
            ANTENNA2, startrow=lrow, nrow=urow-lrow)

        return antenna2.reshape(context.shape)

    @cache_ms_read
    def observed_vis(self, context):
        lrow, urow = context.dim_extents('nrows')

        data = self._tables[ORDERED_MAIN_TABLE].getcol(
            DATA, startrow=lrow, nrow=urow-lrow)

        return data.reshape(context.shape)

    @cache_ms_read
    def flag(self, context):
        lrow, urow = context.dim_extents('nrows')

        flag = self._tables[ORDERED_MAIN_TABLE].getcol(
            FLAG, startrow=lrow, nrow=urow-lrow)

        return flag.reshape(context.shape)

    @cache_ms_read
    def weight(self, context):
        lrow, urow = context.dim_extents('nrows')
        nchan = context.dim_extent_size('nchanperband')

        weight = self._tables[ORDERED_MAIN_TABLE].getcol(
            WEIGHT, startrow=lrow, nrow=urow-lrow)

        # WEIGHT is applied across all channels
        weight = np.repeat(weight, nchan, 0)
        return weight.reshape(context.shape)

    def clear_cache(self):
        self._cache.clear()

    def close(self):
        self.clear_cache()

        for table in self._tables.itervalues():
            table.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()

def test():
    import copy
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

    def feed(feeder, cube, array_names):
        for dims in cube.dim_iter(*dim_iter_args, update_local_size=True):
            cube.update_dimensions(dims)
            cube.update_row_dimensions()
            array_schemas = cube.arrays(reify=True)

            feed_contexts = ((n, FeedContext(n, cube, {},
                array_schemas[n].shape, array_schemas[n].dtype))
                for n in array_names)

            feed_arrays = ((n, getattr(feeder, n)(c)) for n, c in feed_contexts)

            print ' '.join(['{n} {s}'.format(n=n,s=a.shape) for n, a in feed_arrays])

    start = time.clock()
    feed(feeder, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    #feeder.clear_cache()

    start = time.clock()
    feed(feeder, cube, array_names)
    print '{s}'.format(s=time.clock() - start)

    array_names = ('antenna1', 'antenna2', 'uvw',
            'observed_vis', 'flag', 'weight')

    cube = copy.deepcopy(feeder.mscube)
    array_schemas = cube.arrays(reify=True)
    arrays = { a: np.zeros(s.shape, s.dtype) for (a, s) in
        ((a, array_schemas[a]) for a in array_names) }

    print [(k, a.shape) for k, a in arrays.iteritems()]

    feeder = NumpyRimeDataFeeder(arrays, cube)
    feed(feeder, cube, array_names)

if __name__ == '__main__':
    test()
