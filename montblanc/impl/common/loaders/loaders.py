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

import os

import numpy as np
import pyrap.tables as pt

import montblanc
import montblanc.util as mbu

from montblanc.api.loaders import BaseLoader

ANTENNA_TABLE = 'ANTENNA'
SPECTRAL_WINDOW = 'SPECTRAL_WINDOW'
FIELD_TABLE = 'FIELD'
POL_TABLE = 'POLARIZATION'

class MeasurementSetLoader(BaseLoader):
    LOG_PREFIX = 'LOADER:'

    def __init__(self, msfile, auto_correlations=False):
        super(MeasurementSetLoader, self).__init__()

        self.tables = {}
        self.msfile = msfile
        self.antfile = '::'.join((self.msfile, ANTENNA_TABLE))
        self.freqfile = '::'.join((self.msfile, SPECTRAL_WINDOW))
        self.fieldfile = '::'.join((self.msfile, FIELD_TABLE))
        self.polfile = '::'.join((self.msfile, POL_TABLE))

        montblanc.log.info("{lp} Opening Measurement Set {ms}.".format(
            lp=self.LOG_PREFIX, ms=self.msfile))

        # Open the main table
        ms = pt.table(self.msfile, ack=False)

        # Hard-code the field ID for now
        field_id = 0

        # Create a view over the MS, ordered by
        # (1) time (TIME)
        # (2) baseline (ANTENNA1, ANTENNA2)
        # (3) band (SPECTRAL_WINDOW_ID via DATA_DESC_ID)
        ordering_query = ' '.join(["SELECT FROM $ms",
            #"WHERE ANTENNA1 != ANTENNA2",
            "WHERE FIELD_ID={fid}".format(fid=field_id),
            "" if auto_correlations else "AND ANTENNA1 != ANTENNA2",
            "ORDERBY TIME, ANTENNA1, ANTENNA2, "
            "[SELECT SPECTRAL_WINDOW_ID FROM ::DATA_DESCRIPTION][DATA_DESC_ID]"])

        ordered_ms = pt.taql(ordering_query)

        # Open main and sub-tables
        self.tables['main'] = ordered_ms
        self.tables['ant']  = at = pt.table(self.antfile, ack=False, readonly=True)
        self.tables['freq'] = ft = pt.table(self.freqfile, ack=False, readonly=True)
        self.tables['field'] = fit = pt.table(self.fieldfile, ack=False, readonly=True)
        self.tables['pol'] = polt = pt.table(self.polfile, ack=False, readonly=True)

        self.nrows = ordered_ms.nrows()
        self.na = at.nrows()
        # Count distinct timesteps in the MS
        t_query = "SELECT FROM $ordered_ms ORDERBY UNIQUE TIME"
        self.ntime = pt.taql(t_query).nrows()
        # Count number of baselines in the MS
        bl_query = "SELECT FROM $ordered_ms ORDERBY UNIQUE ANTENNA1, ANTENNA2"
        self.nbl = pt.taql(bl_query).nrows()

        # Number of polarizations (assuming to be the same for all spectral windows)
        self.npol = polt.getcol('NUM_CORR')[0]

        # Number of channels per band
        chan_per_band = ft.getcol('NUM_CHAN')

        # Require the same number of channels per band
        if not all(chan_per_band[0] == cpb for cpb in chan_per_band):
            raise ValueError('Channels per band {cpb} are not equal!'
                .format(cpb=chan_per_band))

        # Number of channels equal to sum of channels per band
        self.nbands = len(chan_per_band)
        self.nchan = sum(chan_per_band)

        # Do some logging
        expected_nbl = mbu.nr_of_baselines(self.na, auto_correlations)
        autocor_str = ('with auto-correlations' if auto_correlations
            else 'without auto-correlations')
        autocor_eq = '({na}x{na1})/2'.format(na=self.na, na1=(
            self.na+1 if auto_correlations else self.na-1))

        self.log("Found {na} antenna(s) in the ANTENNA sub-table.".format(
            na=self.na))
        self.log("Found {nbl} of {ebl}={aeq} possible baseline(s) ({astr}).".format(
            aeq=autocor_eq, ebl=expected_nbl, astr=autocor_str, nbl=self.nbl))
        self.log("Found {nb} band(s), containing {cpb} channels.".format(
            nb=self.nbands, nc=chan_per_band[0], cpb=chan_per_band))
        self.log("Found {npol} polarization(s) in the POLARIZATION table.".format(
            npol=self.npol))

        # Sanity check computed rows vs actual rows
        computed_rows = self.ntime*self.nbl*self.nbands

        if not computed_rows == self.nrows:
            montblanc.log.warn("{nt} x {nbl} x {nb} = {cr} does not equal "
                "the number of measurement set rows '{msr}'."
                    .format(nt=self.ntime,
                        nbl=self.nbl, nb=self.nbands,
                        cr=computed_rows, msr=self.nrows))


    def get_dims(self):
        """
        Returns a tuple with the number of
        timesteps, baselines, antenna, channel bands and channels
        """
        # Determine the problem dimensions
        return self.ntime, self.nbl, self.na, self.nbands, self.nchan, self.npol

    def log(self, msg, *args, **kwargs):
        montblanc.log.info('{lp} {m}'.format(lp=self.LOG_PREFIX, m=msg),
            *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Close all the tables
        for table in self.tables.itervalues():
            table.close()
