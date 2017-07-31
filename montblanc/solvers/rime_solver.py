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

import numpy as np
import types

try:
    from inspect import signature
except ImportError:
    from funcsigs import signature

import montblanc
from hypercube import HyperCube

import montblanc.util as mbu

class RIMESolver(object):
    def __init__(self, slvr_cfg):
        """
        RIMESolver Constructor

        Arguments
        ---------
            slvr_cfg : dictionary
                Contains configuration options for this solver
        """
        self._slvr_cfg = slvr_cfg

        # Configure our floating point and complex types
        if slvr_cfg['dtype'] == 'float':
            self.ft = np.float32
            self.ct = np.complex64
        elif slvr_cfg['dtype'] == 'double':
            self.ft = np.float64
            self.ct = np.complex128
        else:
            raise TypeError('Invalid dtype %s ' % slvr_cfg['dtype'])

        # Maintain a hypercube
        self._cube = HyperCube()

        # Is this solver handling auto-correlations
        self._is_auto_correlated = slvr_cfg['auto_correlations']

    @property
    def hypercube(self):
        return self._cube

    def is_float(self):
        return self.ft == np.float32

    def is_double(self):
        return self.ft == np.float64

    def is_autocorrelated(self):
        """ Does this solver handle autocorrelations? """
        return self._is_auto_correlated == True

    def type_dict(self):
        """ Returns a dictionary mapping strings to concrete types """
        return {
            'ft' : self.ft,
            'ct' : self.ct,
            'int' : int,
        }

    def template_dict(self):
        """
        Returns a dictionary suitable for templating strings with
        properties and dimensions related to this Solver object.

        Used in templated GPU kernels.
        """
        slvr = self

        D = {
            # Constants
            'LIGHTSPEED': montblanc.constants.C,
        }

        # Map any types
        D.update(self.type_dict())

        # Update with dimensions
        D.update(self.dim_local_size_dict())

        # Add any registered properties to the dictionary
        for p in self._properties.itervalues():
            D[p.name] = getattr(self, p.name)

        return D

    def config(self):
        """ Returns the configuration dictionary for this solver """
        return self._slvr_cfg

    def solve(self):
        """ Solve the RIME """
        pass

    def initialise(self):
        """ Initialise the RIME solver """
        pass

    def shutdown(self):
        """ Stop the RIME solver """
        pass

    def __enter__(self):
        self.initialise()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()