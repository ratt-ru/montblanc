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

        main_table = pt.table(self.msfile, ack=False)

        # If requested, use TAQL to ignore auto-correlations
        if not auto_correlations:
            self.tables['main'] = main_table.query('ANTENNA1 != ANTENNA2')
        else:
            self.tables['main'] = main_table

        self.tables['ant']  = pt.table(self.antfile, ack=False)
        self.tables['freq'] = pt.table(self.freqfile, ack=False)

    def get_dims(self, auto_correlations=False):
        """
        Returns a tuple with the number of timesteps, antenna and channels
        """
        # Determine the problem dimensions
        na = self.tables['ant'].nrows()
        nbl = mbu.nr_of_baselines(na, auto_correlations)
        nchan = np.asscalar(self.tables['freq'].getcol('NUM_CHAN'))
        ntime = self.tables['main'].nrows() // nbl

        return ntime, na, nchan

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # Close all the tables
        for table in self.tables.itervalues():
            table.close()
