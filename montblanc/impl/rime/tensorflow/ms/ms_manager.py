#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import collections

import numpy as np

import montblanc

from hypercube import HyperCube
import pyrap.tables as pt

# Map MS column string types to numpy types
MS_TO_NP_TYPE_MAP = {
    'INT' : np.int32,
    'FLOAT' : np.float32,
    'DOUBLE' : np.float64,
    'BOOLEAN' : np.bool,
    'COMPLEX' : np.complex64,
    'DCOMPLEX' : np.complex128
}

# Key names for main and taql selected tables
MAIN_TABLE = 'MAIN'
ORDERED_MAIN_TABLE = 'ORDERED_MAIN'
ORDERED_UVW_TABLE = 'ORDERED_UVW'
ORDERED_TIME_TABLE = 'ORDERED_TIME'
ORDERED_BASELINE_TABLE = 'ORDERED_BASELINE'

# Measurement Set sub-table name string constants
ANTENNA_TABLE = 'ANTENNA'
SPECTRAL_WINDOW_TABLE = 'SPECTRAL_WINDOW'
DATA_DESCRIPTION_TABLE = 'DATA_DESCRIPTION'
POLARIZATION_TABLE = 'POLARIZATION'
FIELD_TABLE = 'FIELD'

SUBTABLE_KEYS = (ANTENNA_TABLE,
    SPECTRAL_WINDOW_TABLE,
    DATA_DESCRIPTION_TABLE,
    POLARIZATION_TABLE,
    FIELD_TABLE)

# Main MS column name constants
TIME = 'TIME'
ANTENNA1 = 'ANTENNA1'
ANTENNA2 = 'ANTENNA2'
UVW = 'UVW'
DATA = 'DATA'
FLAG = 'FLAG'
WEIGHT = 'WEIGHT'
MODEL_DATA = 'MODEL_DATA'
CORRECTED_DATA = 'CORRECTED_DATA'

# Antenna sub-table column name constants
POSITION = 'POSITION'

# Field sub-table column name constants
PHASE_DIR = 'PHASE_DIR'

# Spectral window sub-table column name constants
CHAN_FREQ = 'CHAN_FREQ'
NUM_CHAN='NUM_CHAN'
REF_FREQUENCY = 'REF_FREQUENCY'

# Columns used in select statement
SELECTED = [TIME, ANTENNA1, ANTENNA2, UVW,
    DATA, MODEL_DATA, CORRECTED_DATA, FLAG, WEIGHT]

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

UPDATE_DIMENSIONS = ['ntime', 'nbl', 'na', 'nchan', 'nbands', 'npol',
    'npolchan', 'nvis']

# Main measurement set ordering dimensions
MS_DIM_ORDER = ('ntime', 'nbl', 'nbands')
# UVW measurement set ordering dimensions
UVW_DIM_ORDER = ('ntime', 'nbl')


def orderby_clause(dimensions, unique=False):
    columns = ", ".join(m.orderby for m
        in MS_ROW_MAPPINGS if m.dimension in dimensions)

    return " ".join(("ORDERBY", "UNIQUE" if unique else "", columns))

def subtable_name(msname, subtable=None):
    return '::'.join((msname, subtable)) if subtable else msname

def open_table(msname, subtable=None):
    return pt.table(subtable_name(msname, subtable),
        ack=False, readonly=False)

def row_extents(cube, dim_order=None):
    if dim_order is None:
        dim_order = MS_DIM_ORDER

    shape = cube.dim_global_size(*dim_order)
    lower = cube.dim_lower_extent(*dim_order)
    upper = tuple(u-1 for u in cube.dim_upper_extent(*dim_order))

    return (np.ravel_multi_index(lower, shape),
        np.ravel_multi_index(upper, shape) + 1)

def uvw_row_extents(cube):
    return row_extents(cube, UVW_DIM_ORDER)

class MeasurementSetManager(object):
    def __init__(self, msname, slvr_cfg):
        super(MeasurementSetManager, self).__init__()

        self._msname = msname
        # Create dictionary of tables
        self._tables = { k: open_table(msname, k) for k in SUBTABLE_KEYS }

        if not pt.tableexists(msname):
            raise ValueError("'{ms}' does not exist "
                "or is not a Measurement Set!".format(ms=msname))

        # Add imaging columns, just in case
        pt.addImagingColumns(msname, ack=False)

        # Open the main measurement set
        ms = open_table(msname)

        # Access individual tables
        ant, spec, ddesc, pol, field = (self._tables[k] for k in SUBTABLE_KEYS)

        # Sanity check the polarizations
        if pol.nrows() > 1:
            raise ValueError("Multiple polarization configurations!")

        self._npol = npol = pol.getcol('NUM_CORR')[0]

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
        self._auto_correlations = auto_correlations = slvr_cfg['auto_correlations']
        self._field_id = field_id = 0

        # Create a view over the MS, ordered by
        # (1) time (TIME)
        # (2) baseline (ANTENNA1, ANTENNA2)
        # (3) band (SPECTRAL_WINDOW_ID via DATA_DESC_ID)
        ordering_query = " ".join((
            "SELECT FROM $ms",
            "WHERE FIELD_ID={fid}".format(fid=field_id),
            "" if auto_correlations else "AND ANTENNA1 != ANTENNA2",
            orderby_clause(MS_DIM_ORDER)
        ))

        # Ordered Measurement Set
        oms = pt.taql(ordering_query)

        montblanc.log.debug("MS ordering query is '{o}'."
            .format(o=ordering_query))

        # Measurement Set ordered by unique time and baseline
        otblms = pt.taql("SELECT FROM $oms {c}".format(
            c=orderby_clause(UVW_DIM_ORDER, unique=True)))

        # Store the main table
        self._tables[MAIN_TABLE] = ms
        self._tables[ORDERED_MAIN_TABLE] = oms
        self._tables[ORDERED_UVW_TABLE] = otblms

        self._column_descriptors = {col: ms.getcoldesc(col) for col in SELECTED}

        # Count distinct timesteps in the MS
        t_orderby = orderby_clause(['ntime'], unique=True)
        t_query = "SELECT FROM $otblms {c}".format(c=t_orderby)
        self._tables[ORDERED_TIME_TABLE] = ot = pt.taql(t_query)
        self._ntime = ntime = ot.nrows()

        # Count number of baselines in the MS
        bl_orderby = orderby_clause(['nbl'], unique=True)
        bl_query = "SELECT FROM $otblms {c}".format(c=bl_orderby)
        self._tables[ORDERED_BASELINE_TABLE] = obl = pt.taql(bl_query)
        self._nbl = nbl = obl.nrows()

        # Number of channels per band
        self._nchanperband = chan_per_band[0]

        self._nchan = nchan = sum(chan_per_band)
        self._nbands = nbands = len(chan_per_band)
        self._npolchan = npolchan = npol*nchan
        self._nvis = nvis = ntime*nbl*nchan

        # Update the cube with dimension information
        # obtained from the MS
        updated_sizes = [ntime, nbl, ant.nrows(),
            sum(chan_per_band), len(chan_per_band), npol,
            npolchan, nvis]

        self._dim_sizes = dim_sizes = { dim: size for dim, size
            in zip(UPDATE_DIMENSIONS, updated_sizes) }

        shape = tuple(dim_sizes[d] for d in MS_DIM_ORDER)
        expected_rows = np.product(shape)

        if not expected_rows == oms.nrows():
            dim_desc = ", ".join('(%s,%s)' % (d, s) for
                d, s in zip(MS_DIM_ORDER, shape))
            row_desc = " x ".join('%s' % s for s in shape)

            montblanc.log.warn("Encountered '{msr}' rows in '{ms}' "
                "but expected '{rd} = {er}' after finding the following "
                "dimensions by inspection: [{d}]. Irregular Measurement Sets "
                "are not fully supported due to the generality of the format.".format(
                    msr=oms.nrows(), ms=msname,
                    er=expected_rows, rd=row_desc, d=dim_desc))

    def close(self):
        # Close all the tables
        for table in self._tables.itervalues():
            table.close()

    @property
    def msname(self):
        return self._msname

    @property
    def column_descriptors(self):
        return self._column_descriptors

    @property
    def channels_per_band(self):
        return self._nchanperband

    def updated_dimensions(self):
        return [(k, v) for k, v in self._dim_sizes.iteritems()]

    @property
    def auto_correlations(self):
        return self._auto_correlations

    @property
    def field_id(self):
        return self._field_id

    @property
    def main_table(self):
        return self._tables[MAIN_TABLE]

    @property
    def ordered_main_table(self):
        return self._tables[ORDERED_MAIN_TABLE]

    @property
    def ordered_uvw_table(self):
        return self._tables[ORDERED_UVW_TABLE]

    @property
    def ordered_time_table(self):
        return self._tables[ORDERED_TIME_TABLE]

    @property
    def antenna_table(self):
        return self._tables[ANTENNA_TABLE]

    @property
    def spectral_window_table(self):
        return self._tables[SPECTRAL_WINDOW_TABLE]

    @property
    def data_description_table(self):
        return self._tables[DATA_DESCRIPTION_TABLE]

    @property
    def polarization_table(self):
        return self._tables[POLARIZATION_TABLE]

    @property
    def field_table(self):
        return self._tables[FIELD_TABLE]

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        self.close()
