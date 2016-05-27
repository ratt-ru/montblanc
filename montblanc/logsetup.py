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

import logging
import logging.handlers

def setup_logging():
    """ Setup logging configuration """

    # Console formatter, mention name
    cfmt = logging.Formatter(('%(name)s - %(levelname)s - %(message)s'))

    # File formatter, mention time
    ffmt = logging.Formatter(('%(asctime)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    # File handler
    fh = logging.handlers.RotatingFileHandler('montblanc.log',
        maxBytes=10*1024*1024, backupCount=10)
    fh.setLevel(logging.INFO)
    fh.setFormatter(ffmt)

    # Create the logger,
    # adding the console and file handler
    mb_logger = logging.getLogger('montblanc')
    mb_logger.handlers = []
    mb_logger.setLevel(logging.DEBUG)
    mb_logger.addHandler(ch)
    mb_logger.addHandler(fh)

    # Set up the concurrent.futures logger
    cf_logger = logging.getLogger('concurrent.futures')
    cf_logger.setLevel(logging.DEBUG)
    cf_logger.addHandler(ch)
    cf_logger.addHandler(fh)

    return mb_logger

def setup_test_logging():
    # Console formatter, mention name
    cfmt = logging.Formatter(('%(name)s - %(levelname)s - %(message)s'))

    # File formatter, mention time
    ffmt = logging.Formatter(('%(asctime)s - %(levelname)s - %(message)s'))

    # Only warnings and more serious stuff on the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    ch.setFormatter(cfmt)

    # Outputs DEBUG level logging to file
    fh = logging.FileHandler('test.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(ffmt)

    # Set up the montblanc logger
    mb_logger = logging.getLogger('montblanc')
    mb_logger.handlers = []
    mb_logger.setLevel(logging.DEBUG)
    mb_logger.addHandler(ch)
    mb_logger.addHandler(fh)

    # Set up the concurrent.futures logger
    cf_logger = logging.getLogger('concurrent.futures')
    cf_logger.setLevel(logging.DEBUG)
    cf_logger.addHandler(ch)
    cf_logger.addHandler(fh)

    return mb_logger