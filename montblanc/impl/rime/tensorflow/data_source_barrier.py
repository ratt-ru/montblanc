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

class DataSourceBarrier(object):
    """
    Simple barrier object that invokes a callback when all
    the required data associated with a key is present

    .. code-block:: python

        def handle_data(key, entry):
            for k, v in entry.iteritems():
                print key, k, v

        # Each dictionary entry should have
        # 'model_vis', 'uvw' and 'descriptor' keys
        barrier = DataSourceBarrier(
                ['model_vis', 'uvw', 'descriptor'],
                handle_data)

        # Store two of the three required data
        # for key 'bar'
        barrier.store('bar', 'model_vis', 100)
        barrier.store('bar', 'uvw', 10)

        # Store two of the three required data
        # for key 'foo'
        barrier.store('foo', 'model_vis', 100)
        barrier.store('foo', 'uvw', 10)

        # Store the entries two for keys 'foo' and 'bar'
        # The handle_data callback is first invoked for 'foo'
        barrier.store('foo', 'descriptor', 'foo')
        barrier.store('bar', 'descriptor', 'bar')
    """
    def __init__(self, data_keys, callback):
        """
        Construct the DataSinkBarrier

        Parameters
        ----------
        data_keys : list
            A list of keys that should be present in each entry
        callback : callable
            Callable of the form :code:`def handle_data(key, entry)`
        """

        self._data_keys = frozenset(data_keys)
        self._data_store = {}
        self._callback = callback
        self._lock = threading.Lock()

    def store(self, key, data_key, data=None):
        """
        Store data in the barrier in the entry associated with key

        Parameters
        ----------
        key : object
            Outer key
        data_key : dict or list or object
            If a dictionary, this will be used to update the inner entry
            If a list, a dict will be created by zipping with data_keys
            passed through to the constructor and used to update
            the inner entry.
            Otherwise, this will be the key use to store data in the
            inner entry
        data : object
            Data associated with data_key, if data_key is not a
            dict or list
        """

        if isinstance(data_key, dict) and data is None:
            data_dict = data_key
        else:
            data_dict = {data_key: data}

        # Guard access with a lock
        with self._lock:
            # Obtain a possibly empty entry for this key
            entry = self._data_store.setdefault(key, {})
            entry.update(data_dict)

            # If our entry has all the required keys,
            # we can invoke the callback
            intersection = set(entry.keys()).intersection(self._data_keys)
            entry_full = intersection == self._data_keys

            # First remove it from the data source
            if entry_full:
                del self._data_store[key]

        if entry_full:
            self._callback(key, entry)

import unittest

class BarrierTest(unittest.TestCase):
    def test_barrier(self):
        def handle_data(key, entry):
            for k, v in entry.iteritems():
                print key, k, v

        # Each dictionary entry should have
        # 'model_vis', 'uvw' and 'descriptor' keys
        barrier = DataSourceBarrier(
                ['model_vis', 'uvw', 'descriptor'],
                handle_data)

        # Store two of the three required data
        # for key 'bar'
        barrier.store('bar', 'model_vis', 100)
        barrier.store('bar', 'uvw', 10)

        # Store two of the three required data
        # for key 'foo'
        barrier.store('foo', 'model_vis', 100)
        barrier.store('foo', 'uvw', 10)

        # Store the entries two for keys 'foo' and 'bar'
        # The handle_data callback is first invoked for 'foo'
        barrier.store('foo', 'descriptor', 'foo')
        barrier.store('bar', 'descriptor', 'bar')

        barrier.store('foo', {'model_vis': 100, 'uvw': 10})
        barrier.store('foo', {'descriptor': 'qux'})

if __name__ == "__main__":
    unittest.main()
