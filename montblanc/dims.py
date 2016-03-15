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

from attrdict import AttrDict
import numpy as np

from montblanc.src_types import SOURCE_VAR_TYPES
from montblanc.enums import DIMDATA

DEFAULT_DESCRIPTION = 'The FOURTH dimension!'

def create_dim_data(name, dim_data, **kwargs):
    return Dimension(name, dim_data, **kwargs)

class Dimension(AttrDict):
    def __init__(self, name, dim_data, **kwargs):
        """
        Create a dimension data dictionary from dim_data
        and keyword arguments. Keyword arguments will be
        used to update the dictionary.

        Arguments
        ---------
            name : str
                Name of the dimension
            dim_data : integer or another Dimension
                If integer a fresh dictionary will be created, otherwise
                dim_data will be copied.

        Keyword Arguments
        -----------------
            Any keyword arguments applicable to the Dimension
            object will be applied at the end of the construction process.
        """
        super(Dimension, self).__init__()

        # If dim_data is an integer, start constructing a dictionary from it
        if isinstance(dim_data, (int, long, np.integer)):
            self[DIMDATA.NAME] = name
            self[DIMDATA.GLOBAL_SIZE] = dim_data
            self[DIMDATA.LOCAL_SIZE] = dim_data
            self[DIMDATA.EXTENTS] = [0, dim_data]
            self[DIMDATA.DESCRIPTION] = DEFAULT_DESCRIPTION
            self[DIMDATA.SAFETY] = True
            self[DIMDATA.ZERO_VALID] = False
        # Otherwise directly copy the entries
        elif isinstance(dim_data, Dimension):
            for k, v in dim_data.iteritems():
                self[k] = v
            self[DIMDATA.NAME] = name
        else:
            raise TypeError(("dim_data must be an integer or a Dimension. "
                "Received a {t} instead.").format(t=type(dim_data)))

        # Intersect the keyword arguments with dictionary values
        kwargs = {k: kwargs[k] for k in kwargs.iterkeys() if k in DIMDATA.ALL}

        # Now update the dimension data from any keyword arguments
        self.update(kwargs)

    def copy(self):
        """ Defer to the constructor for copy operations """
        return Dimension(self[DIMDATA.NAME], self)

    def update(self, other=None, **kwargs):
        """
        Sanitised dimension data update
        """

        from collections import Mapping, Sequence

        # Just pack everything from other into kwargs
        # for the updates below
        # See http://stackoverflow.com/a/30242574
        if other is not None:
            for k, v in other.iteritems() if isinstance(other, Mapping) else other:
                kwargs[k] = v

        if DIMDATA.NAME in kwargs:
            self[DIMDATA.NAME] = kwargs[DIMDATA.NAME]

        name = self[DIMDATA.NAME]

        if DIMDATA.DESCRIPTION in kwargs:
            self[DIMDATA.DESCRIPTION] = kwargs[DIMDATA.DESCRIPTION]

        # Update options if present
        if DIMDATA.SAFETY in kwargs:
            self[DIMDATA.SAFETY] = kwargs[DIMDATA.SAFETY]

        if DIMDATA.ZERO_VALID in kwargs:
            self[DIMDATA.ZERO_VALID] = kwargs[DIMDATA.ZERO_VALID]

        if DIMDATA.LOCAL_SIZE in kwargs:
            if self[DIMDATA.SAFETY] is True:
                raise ValueError(("Modifying local size of dimension '{d}' "
                    "is not allowed by default. If you are sure you want "
                    "to do this add a '{s}' : 'False' entry to the "
                    "update dictionary.").format(d=name, s=DIMDATA.SAFETY))

            if self[DIMDATA.ZERO_VALID] is False and kwargs[DIMDATA.LOCAL_SIZE] == 0:
                raise ValueError(("Modifying local size of dimension '{d}' "
                    "to zero is not valid. If you are sure you want "
                    "to do this add a '{s}' : 'True' entry to the "
                    "update dictionary.").format(d=name, s=DIMDATA.ZERO_VALID))

            self[DIMDATA.LOCAL_SIZE] = kwargs[DIMDATA.LOCAL_SIZE]

        if DIMDATA.EXTENTS in kwargs:
            exts = kwargs[DIMDATA.EXTENTS]
            if (not isinstance(exts, Sequence) or len(exts) != 2):
                raise TypeError("'{e}' entry must be a "
                    "sequence of length 2.".format(e=DIMDATA.EXTENTS))

            self[DIMDATA.EXTENTS] = [v for v in exts[0:2]]

        # Check that we've been given valid values
        self.check()

    def check(self):
        """ Sanity check the contents of a dimension data dictionary """
        ls, gs, E, name, zeros = (self[DIMDATA.LOCAL_SIZE],
            self[DIMDATA.GLOBAL_SIZE],
            self[DIMDATA.EXTENTS],
            self[DIMDATA.NAME],
            self[DIMDATA.ZERO_VALID])

        # Sanity check dimensions
        assert 0 <= ls <= gs, \
            ("Dimension '{n}' local size {l} is greater than "
            "it's global size {g}").format(
                n=name, l=ls, g=gs)

        assert E[1] - E[0] <= ls, \
            ("Dimension '{n}' local size {l} is greater than "
            "it's extents [{e0}, {e1}]").format(
                n=name, l=ls, e0=E[0], e1=Ep[1])

        if zeros:
            assert 0 <= E[0] <= E[1] <= gs, (
                "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                    .format(d=name, gs=gs, e0=E[0], e1=E[1]))
        else:
            assert 0 <= E[0] < E[1] <= gs, (
                "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                    .format(d=name, gs=gs, e0=E[0], e1=E[1]))    

