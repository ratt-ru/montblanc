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

class DataSinkBarrier(object):
    """
    Simple barrier object that handles stores

    .. code-block:: python

        # Each dictionary entry should have
        # 'model_vis', 'uvw' and 'descriptor' keys
        barrier = DataSinkBarrier(['model_vis', 'uvw', 'descriptor'])

        # Store using single data_key:data
        barrier.store('bar', 'model_vis', 100)
        barrier.store('bar', 'uvw', 10)
        barrier.store('bar', 'descriptor', 'bar')

        # Store using dictionaries
        barrier.store('foo', {'model_vis': 100, 'uvw': 10})
        barrier.store('foo', {'descriptor': 'qux'})

        # Pop data associated with single data key
        print barrier.pop('foo', 'model_vis')
        print barrier.pop('foo', 'descriptor')
        print barrier.pop('foo', 'uvw')

        # Pop entire entry associated with key
        print barrier.pop('bar')
    """
    def __init__(self, data_keys):
        self._data_keys = data_keys
        self._data_store = {}
        self._cond = threading.Condition(threading.Lock())

    def store(self, key, data_key, data=None):
        """
        Store data in the barrier in the entry for key
        """

        if isinstance(data_key, dict) and data is None:
            data_dict = data_key
        elif isinstance(data_key, list) and data is None:
            data_dict = {n: d for n, d in zip(self._data_keys, data_key)}
        else:
            data_dict = {data_key: data}

        # Guard access with a lock
        with self._cond:
            # Obtain a possibly empty entry for this key
            entry = self._data_store.setdefault(key, {})
            entry.update(data_dict)
            # Notify any getters that a new entry is available
            self._cond.notifyAll()

    def keys(self):
        with self._cond:
            return list(self._data_store.keys())

    def pop(self, key, data_key=None, timeout=None):
        """
        Pops the data mapped to data_key in the entry
        mapped to key. If data_key is None, the
        entire entry is popped.

        Parameters
        ----------
        key : str
            Key associated with the barrier entry
        data_key : str
            Key associated with data inside entry
            associated with key.
        timeout : None or float
            if a floating point value is provided
            this will be used as the time to wait



        Returns
        -------
        object
            The data mapped to data_key in the entry
            mapped to key. If data_key is None, the
            entire entry is returned.
        """

        def _pop(self):
            try:
                entry = self._data_store[key]
            except KeyError as e:
                raise KeyError("'{}' not in barrier".format(key))

            # Return entire entry if data_key is None
            if data_key is None:
                del self._data_store[key]
                return entry

            # Otherwise return the data entry associated
            # with data key, deleting the entry if it
            # becomes empty
            try:
                data_entry = entry.pop(data_key)

                if len(entry) == 0:
                    del self._data_store[key]

                return data_entry
            except KeyError as e:
                raise KeyError("Couldn't pop {}".format(data_key))

        while True:
            with self._cond:
                try:
                    return _pop(self)
                except KeyError:
                    if timeout is None:
                        raise

                    self._cond.wait(timeout)



import unittest

class BarrierTest(unittest.TestCase):
    def test_barrier(self):
        # Each dictionary entry should have
        # 'model_vis', 'uvw' and 'descriptor' keys
        barrier = DataSinkBarrier(['model_vis', 'uvw', 'descriptor'])

        # Store using single data_key:data
        barrier.store('bar', 'model_vis', 100)
        barrier.store('bar', 'uvw', 10)
        barrier.store('bar', 'descriptor', 'bar')

        # Store using dictionaries
        barrier.store('foo', {'model_vis': 100, 'uvw': 10})
        barrier.store('foo', {'descriptor': 'qux'})

        # Pop data associated with single data key
        print barrier.pop('foo', 'model_vis')
        print barrier.pop('foo', 'descriptor')
        print barrier.pop('foo', 'uvw')

        # Pop entire entry associated with key
        print barrier.pop('bar')

if __name__ == "__main__":
    unittest.main()