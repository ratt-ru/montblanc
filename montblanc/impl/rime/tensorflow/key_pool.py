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

import threading

class KeyPool(object):
    """ Pool of reusable integer keys """
    def __init__(self):
        self._keys = []
        self._lock = threading.Lock()
        self._last_key = 0

    def get(self, nkeys):
        """ Returns nkeys keys """
        with self._lock:
            keys = self._keys[0:nkeys]
            del self._keys[0:nkeys]

            remaining = nkeys - len(keys)

            if remaining > 0:
                keys.extend(xrange(self._last_key, self._last_key + remaining))
                self._last_key += remaining

        return keys

    def release(self, keys):
        """ Releases keys back into the pool """
        with self._lock:
            self._keys.extend(keys)

import six
import unittest

class KeyPoolTest(unittest.TestCase):
    def test_key_pool(self):
        keypool = KeyPool()

        keys = keypool.get(10)
        self.assertTrue(keys == list(six.moves.range(10)))

        keys, rel_keys = keys[0:5], keys[5:10]

        keypool.release(rel_keys)

        more_keys = keypool.get(10)
        self.assertTrue(more_keys == list(six.moves.range(5,15)))

if __name__ == "__main__":
    unittest.main()



