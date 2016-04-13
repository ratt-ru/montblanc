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

import argparse
import json
import numpy as np
import os
import re

import montblanc
from montblanc.src_types import default_sources

# Argument
class ArgumentParserError(Exception):
    pass

class ThrowingArgumentParser(argparse.ArgumentParser):
    """
    Inherit from argparse.ArgumentParser and override
    the error message
    """
    _ARG_STR_RE = re.compile('^(?:argument\s*?(\S*?)):\s*?(.*?)$')

    def error(self, message):
        """
        Massage the error message a bit and throw
        an exception, instead of calling the sys.exit
        in the argparse.ArgumentParser parent class
        """
        m = self._ARG_STR_RE.match(message)

        if not m:
            raise ArgumentParserError(message)

        arg_name = m.group(1).strip('-')

        if len(arg_name) > 0:
            raise ArgumentParserError(
                "argument '{n}': {m}".format(
                    n=arg_name, m=m.group(2)))

        raise ArgumentParserError(message)

class SolverConfig(object):
    CFG_FILE = 'cfg_file'
    CFG_FILE_DESCRIPTION = 'Configuration File'

    MODE = 'mode'
    MODE_CHI_SQUARED = 'chi-squared'
    MODE_SIMULATOR = 'simulator'
    DEFAULT_MODE = MODE_CHI_SQUARED
    MODE_DESCRIPTION = (
        "Montblanc's execution mode. "
        "If '{chi}', montblanc will compute model visibilities and "
        "use them in conjunction with observed visibilities, "
        "flag data and weighting vectors to compute a chi-squared value. "
        "If '{sim}', montblanc will compute model visibilities.").format(
            chi=MODE_CHI_SQUARED, sim=MODE_SIMULATOR)

    SOURCES = 'sources'
    DEFAULT_SOURCES = {k: v for k, v
            in default_sources().iteritems()}
    SOURCES_DESCRIPTION = (
        "Dictionary containing source type numbers "
        "e.g. {d}").format(d=DEFAULT_SOURCES)

    # Number of timesteps
    NTIME = 'ntime'
    DEFAULT_NTIME = 10
    NTIME_DESCRIPTION = 'Timesteps'

    # Number of antenna
    NA = 'na'
    DEFAULT_NA = 7
    NA_DESCRIPTION = 'Antenna'

    # Number of baselines
    NBL = 'nbl'
    NBL_DESCRIPTION = 'Baselines'

    # Number of channels
    NCHAN = 'nchan'
    DEFAULT_NCHAN = 16
    NCHAN_DESCRIPTION = 'Channels'

    # Number of polarisations
    NPOL = 'npol'
    DEFAULT_NPOL = 4
    NPOL_DESCRIPTION = 'Polarisations'

    # Number of sources
    NSRC = 'nsrc'
    NSRC_DESCRIPTION = 'Sources (total)'

    # Master solver
    SOLVER_TYPE = 'solver_type'
    SOLVER_TYPE_MASTER = 'master'
    SOLVER_TYPE_SLAVE = 'slave'
    DEFAULT_SOLVER_TYPE = SOLVER_TYPE_MASTER
    VALID_SOLVER_TYPE = [SOLVER_TYPE_MASTER, SOLVER_TYPE_SLAVE]
    SOLVER_TYPE_DESCRIPTION = (
        'String indicating the solver type. '
        "If '{m}' it controls other solvers "
        "If '{s}' it is controlled by master solvers ").format(
            m=SOLVER_TYPE_MASTER, s=SOLVER_TYPE_SLAVE)

    # Are we dealing with floats or doubles?
    DTYPE = 'dtype'
    DTYPE_FLOAT = 'float'
    DTYPE_DOUBLE = 'double'
    DEFAULT_DTYPE = DTYPE_DOUBLE
    VALID_DTYPES = [DTYPE_FLOAT, DTYPE_DOUBLE]
    DTYPE_DESCRIPTION = (
        'Type of floating point precision used to compute the RIME. ' 
        "If '{f}', compute the RIME with single-precision "
        "If '{d}', compute the RIME with double-precision.").format(
            f=DTYPE_FLOAT, d=DTYPE_DOUBLE)

    # Should we handle auto correlations
    AUTO_CORRELATIONS = 'auto_correlations'
    DEFAULT_AUTO_CORRELATIONS = False
    VALID_AUTO_CORRELATIONS = [True, False]
    AUTO_CORRELATIONS_DESCRIPTION = ('Take auto-correlations into account '
        'when computing number of baselines from number antenna.')

    # Data Source. Defaults/A MeasurementSet/Random Test data
    DATA_SOURCE = 'data_source'
    DATA_SOURCE_DEFAULT = 'default'
    DATA_SOURCE_MS = 'ms'
    DATA_SOURCE_TEST = 'test'
    DATA_SOURCE_EMPTY = 'empty'
    DEFAULT_DATA_SOURCE = DATA_SOURCE_MS
    VALID_DATA_SOURCES = [DATA_SOURCE_DEFAULT, DATA_SOURCE_MS,
        DATA_SOURCE_TEST, DATA_SOURCE_EMPTY]
    DATA_SOURCE_DESCRIPTION = (
        "The data source for initialising data arrays. "
        "If '{d}', data is initialised with defaults. " 
        "If '{t}' filled with random test data. "
        "If '{ms}', some data will be read from a MeasurementSet, "
        "and defaults will be used for the rest. "
        "If '{e}', the arrays will not be initialised").format(
            d=DATA_SOURCE_DEFAULT, ms=DATA_SOURCE_MS,
            t=DATA_SOURCE_TEST, e=DATA_SOURCE_EMPTY)

    # MeasurementSet file
    MS_FILE = 'msfile'
    MS_FILE_DESCRIPTION = 'MeasurementSet file'

    DATA_ORDER = 'data_order'
    DATA_ORDER_CASA = 'casa'
    DATA_ORDER_OTHER = 'other'
    DEFAULT_DATA_ORDER = DATA_ORDER_CASA
    VALID_DATA_ORDER = [DATA_ORDER_CASA, DATA_ORDER_OTHER]
    DATA_ORDER_DESCRIPTION = (
        "MeasurementSet data ordering. "
        "If '{c}', assume CASA's default ordering of time x baseline. "
        "If '{o}', assume baseline x time ordering").format(
            c=DATA_ORDER_CASA, o=DATA_ORDER_OTHER)

    CONTEXT = 'context'
    CONTEXT_DESCRIPTION = ('PyCUDA context(s) '
        'available for this solver to use. '
        'Should be of type pycuda.driver.Context. '
        'May be a single context of a list of contexts')

    DESCRIPTION = 'description'
    DEFAULT = 'default'
    VALID = 'valid'
    REQUIRED = 'required'

    descriptions = {
        MODE : {
            DESCRIPTION: MODE_DESCRIPTION,
            DEFAULT: DEFAULT_MODE,
            REQUIRED: True },

        SOURCES: {
            DESCRIPTION: SOURCES_DESCRIPTION,
            DEFAULT: DEFAULT_SOURCES,
            REQUIRED: True },

        NTIME: {
            DESCRIPTION: NTIME_DESCRIPTION,
            DEFAULT: DEFAULT_NTIME,
            REQUIRED: True },

        NA: {
            DESCRIPTION: NA_DESCRIPTION,
            DEFAULT: DEFAULT_NA,
            REQUIRED: True },

        NBL: {
            DESCRIPTION: NBL_DESCRIPTION,
            REQUIRED: False },

        NCHAN: {
            DESCRIPTION: NCHAN_DESCRIPTION,
            DEFAULT: DEFAULT_NCHAN,
            REQUIRED: True },

        NPOL: {
            DESCRIPTION: NPOL_DESCRIPTION,
            DEFAULT: DEFAULT_NPOL,
            REQUIRED: True },

        DTYPE: {
            DESCRIPTION: DTYPE_DESCRIPTION,
            DEFAULT: DEFAULT_DTYPE,
            VALID: VALID_DTYPES,
            REQUIRED: True },

        AUTO_CORRELATIONS: {
            DESCRIPTION: AUTO_CORRELATIONS_DESCRIPTION,
            DEFAULT: DEFAULT_AUTO_CORRELATIONS,
            VALID: VALID_AUTO_CORRELATIONS
        },

        DATA_SOURCE: {
            DESCRIPTION: DATA_SOURCE_DESCRIPTION,
            DEFAULT: DEFAULT_DATA_SOURCE,
            VALID: VALID_DATA_SOURCES,
            REQUIRED: True
        },

        MS_FILE: {
            DESCRIPTION:  MS_FILE_DESCRIPTION,
        },

        DATA_ORDER: {
            DESCRIPTION: DATA_ORDER_DESCRIPTION,
            DEFAULT: DEFAULT_DATA_ORDER,
            VALID: VALID_DATA_ORDER,
            REQUIRED: True
        },

        CONTEXT : {
            DESCRIPTION: CONTEXT_DESCRIPTION,
            REQUIRED: True
        },

        SOLVER_TYPE : {
            DESCRIPTION: SOLVER_TYPE_DESCRIPTION,
            DEFAULT: DEFAULT_SOLVER_TYPE,
            REQUIRED: True
        },
    }

    def parser(self):
        """
        Returns an argparse parser suitable for parsing command line options,
        but employed here to manage the Solver Configuration options defined above.
        """
        p = ThrowingArgumentParser()

        p.add_argument('--{v}'.format(v=self.MODE),
            required=False,
            type=str,
            help=self.MODE_DESCRIPTION,
            default=self.DEFAULT_MODE)

        p.add_argument('--{v}'.format(v=self.SOURCES),
            required=False,
            type=json.loads,
            help=self.SOURCES_DESCRIPTION,
            default=json.dumps(self.DEFAULT_SOURCES))

        p.add_argument('--{v}'.format(v=self.NTIME),
            required=False,
            type=int,
            help=self.NTIME_DESCRIPTION,
            default=self.DEFAULT_NTIME)

        p.add_argument('--{v}'.format(v=self.NA),
            required=False,
            type=int,
            help=self.NA_DESCRIPTION,
            default=self.DEFAULT_NA)

        p.add_argument('--{v}'.format(v=self.NPOL),
            required=False,
            type=int,
            help=self.NPOL_DESCRIPTION,
            default=self.DEFAULT_NPOL)

        p.add_argument('--{v}'.format(v=self.NCHAN),
            required=False,
            type=int,
            help=self.NCHAN_DESCRIPTION,
            default=self.DEFAULT_NCHAN)

        p.add_argument('--{v}'.format(v=self.SOLVER_TYPE),
            required=False,
            type=str,
            help=self.SOLVER_TYPE_DESCRIPTION,
            choices=self.VALID_SOLVER_TYPE,
            default=self.SOLVER_TYPE_MASTER)

        p.add_argument('--{v}'.format(v=self.DTYPE),
            required=False,
            type=str,
            choices=self.VALID_DTYPES,
            help=self.DTYPE_DESCRIPTION,
            default=self.DEFAULT_DTYPE)

        p.add_argument('--{v}'.format(v=self.AUTO_CORRELATIONS),
            required=False,
            type=bool,
            choices=self.VALID_AUTO_CORRELATIONS,
            help=self.AUTO_CORRELATIONS_DESCRIPTION,
            default=self.DEFAULT_AUTO_CORRELATIONS)

        p.add_argument('--{v}'.format(v=self.DATA_SOURCE),
            required=False,
            type=str,
            choices=self.VALID_DATA_SOURCES,
            help=self.DATA_SOURCE_DESCRIPTION,
            default=self.DEFAULT_DATA_SOURCE)

        p.add_argument('--{v}'.format(v=self.MS_FILE),
            required=False,
            type=str,
            help=self.MS_FILE_DESCRIPTION)

        p.add_argument('--{v}'.format(v=self.DATA_ORDER),
            required=False,
            type=str,
            choices=self.VALID_DATA_ORDER,
            help=self.DATA_ORDER_DESCRIPTION,
            default=self.DEFAULT_DATA_ORDER)

        p.add_argument('--{v}'.format(v=self.CONTEXT),
            required=False,
            help=self.CONTEXT_DESCRIPTION)

        return p

    def gen_cfg(self, **kwargs):
        """ Generate a configuration dictionary from the supplied kwargs """
        # Get a parser
        p = self.parser()

        # A configuration file was specified
        # Load any montblanc section options into theargument parser
        if self.CFG_FILE in kwargs:
            import ConfigParser
            
            montblanc.log.info("Loading defaults from '{cf}'.".format(
                cf=kwargs[self.CFG_FILE]))
            config = ConfigParser.SafeConfigParser()
            config.read([kwargs[self.CFG_FILE]])
            defaults = dict(config.items('montblanc'))
            p.set_defaults(**defaults)

        # Update defaults with supplied kwargs
        p.set_defaults(**kwargs)
        # Now parse the arguments and get a dictionary
        # representing the solver configuration
        slvr_cfg = vars(p.parse_args([]))

        return slvr_cfg

