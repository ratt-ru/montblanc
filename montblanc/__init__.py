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

def get_montblanc_path():
    """ Return the current path in which montblanc is installed """
    import montblanc
    return os.path.dirname(inspect.getfile(montblanc))

def get_include_path():
    return os.path.join(get_montblanc_path(), 'include')

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

def rime_solver_cfg(**kwargs):
    """
    Produces a SolverConfiguration object, inherited from
    a simple python dict, and containing the options required
    to configure the RIME Solver.

    Keyword arguments
    -----------------
    Any keyword arguments are inserted into the
    returned dict.

    Returns
    -------
    A SolverConfiguration object.
    """
    from configuration import (load_config, config_validator,
        raise_validator_errors)

    def _merge_copy(d1, d2):
        return { k: _merge_copy(d1[k], d2[k]) if k in d1
                                                and isinstance(d1[k], dict)
                                                and isinstance(d2[k], dict)
                                            else d2[k] for k in d2 }

    try:
        cfg_file = kwargs.pop('cfg_file')
    except KeyError as e:
        slvr_cfg = kwargs
    else:
        cfg = load_config(cfg_file)
        slvr_cfg = _merge_copy(cfg, kwargs)

    # Validate the configuration, raising any errors
    validator = config_validator()
    validator.validate(slvr_cfg)
    raise_validator_errors(validator)

    return validator.document

def rime_solver(slvr_cfg):
    """
    rime_solver(slvr_cfg)

    Returns a solver suitable for solving the RIME.

    Parameters
    ----------
    slvr_cfg : RimeSolverConfiguration
            Solver Configuration.

    Returns
    -------
    A solver
    """

    import montblanc.factory

    return montblanc.factory.rime_solver(slvr_cfg)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
