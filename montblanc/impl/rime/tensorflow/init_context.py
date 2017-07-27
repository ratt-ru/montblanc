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

class InitialisationContext(object):
    """
    Initialisation Context object passed to Providers.

    It provides initialisation information to a Provider,
    allowing Providers to perform setup based on
    configuration.

    .. code-block:: python

        class CustomSourceProvider(SourceProvider):
            def init(self, init_context):
                config = context.cfg()
                ...
    """
    __slots__ = ('_cfg',)

    def __init__(self, slvr_cfg):
        self._cfg = slvr_cfg

    @property
    def cfg(self):
        """
        Configuration
        """
        return self._cfg

    def help(self, display_cube=False):
        """
        Get help associated with this context

        Returns
        -------
            str
                A help string associated with this context
        """
        return """ Call context.cfg to access the solver configuration """