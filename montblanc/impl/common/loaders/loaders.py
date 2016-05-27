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

class MeasurementSetLoader(BaseLoader):
    LOG_PREFIX = 'LOADER:'

    def __init__(self, msfile, auto_correlations=False):
        super(MeasurementSetLoader, self).__init__()

        self.tables = {}
        self.msfile = msfile
        self.antfile = '::'.join([self.msfile, ANTENNA_TABLE])
        self.freqfile = '::'.join([self.msfile, SPECTRAL_WINDOW])

        montblanc.log.info("{lp} Opening Measurement Set {ms}.".format(
            lp=self.LOG_PREFIX, ms=self.msfile))

        # Open the main table
        main_table = pt.table(self.msfile, ack=False)

        # If requested, use TAQL to ignore auto-correlations
        if not auto_correlations:
            self.tables['main'] = main_table.query('ANTENNA1 != ANTENNA2')
        else:
            self.tables['main'] = main_table

        # Open antenna and frequency tables
        self.tables['ant']  = pt.table(self.antfile, ack=False)
        self.tables['freq'] = pt.table(self.freqfile, ack=False)

        msrows = self.tables['main'].nrows()
        self.na = self.tables['ant'].nrows()
        self.nbl = mbu.nr_of_baselines(self.na, auto_correlations)
        self.nchan = np.asscalar(self.tables['freq'].getcol('NUM_CHAN'))
        self.ntime = msrows // self.nbl

        # Require a ntime x nbl shape for MS rows
        if msrows != self.ntime*self.nbl:
            autocor_str = ('with auto-correlations' if auto_correlations
                else 'without auto-correlations')

            raise ValueError("{na} antenna {astr} produce {nbl} baselines, "
                "but {msr}, the number of rows in '{msf}', cannot "
                "be divided exactly by this number.".format(
                    na=self.na, nbl=self.nbl, astr=autocor_str,
                    msr=msrows, msf=self.msfile))        

    def get_dims(self):
        """
        Returns a tuple with the number of timesteps, antenna and channels
        """
        # Determine the problem dimensions
        return self.ntime, self.na, self.nchan

    def log(self, msg, *args, **kwargs):
        montblanc.log.info('{lp} {m}'.format(lp=self.LOG_PREFIX, m=msg),
            *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Close all the tables
        for table in self.tables.itervalues():
            table.close()
