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

import inspect
import os

from montblanc.logsetup import setup_logging, setup_test_logging
from montblanc.tests import test
from montblanc.version import __version__

log = setup_logging()

# This solution for constants based on
# http://stackoverflow.com/a/2688086
# Create a property that throws when
# you try and set it
def constant(f):
    def fset(self, value):
        raise SyntaxError, 'Foolish Mortal! You would dare change a universal constant?'
    def fget(self):
        return f()

    return property(fget, fset)

class MontblancConstants(object):
    # The speed of light, in metres
    @constant
    def C():
        return 299792458

# Create a constants object
constants = MontblancConstants()

from montblanc.impl.rime.tensorflow.dask_rime import Rime
from montblanc.impl.rime.tensorflow.dataset import (default_dataset,
    montblanc_dataset, dataset_from_ms, rechunk_to_budget)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
