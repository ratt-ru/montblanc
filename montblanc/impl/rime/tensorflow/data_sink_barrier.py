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

try:
    import threading2 as threading # python 2
except ImportError:
    import threading               # python 3

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
        self._closed = threading.Event()

    def store(self, key, data_key, data=None):
        """
        Store data in the barrier in the entry for key
        """

        if self.closed():
            raise ValueError("Data Barrier is closed")

        if isinstance(data_key, dict) and data is None:
            data_dict = data_key
        elif isinstance(data_key, list) and data is None:
            data_dict = { n: d for n, d in zip(self._data_keys, data_key) }
        else:
            data_dict = { data_key: data }

        # Guard access with a lock
        with self._cond:
            # Obtain a possibly empty entry for this key
            entry = self._data_store.setdefault(key, {})
            entry.update(data_dict)
            # Notify any getters that a new entry is available
            self._cond.notifyAll()

    def keys(self):
        """ Returns barrier entry keys """
        with self._cond:
            return list(self._data_store.keys())

    def __len__(self):
        """ Returns the number of entries in the barrier """
        with self._cond:
            return len(self._data_store)

    def close(self):
        """ Closes the data barrier to stores and gets """
        self._closed.set()

    def closed(self):
        """ Return True if the barrier is closed """
        return self._closed.is_set()

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
        timeout : False or None or float
            if False, this method will not block,
            if float, this method will block for the specified time
            if None, this method will block until data is available

        Returns
        -------
        object
            The data mapped to data_key in the entry
            mapped to key. If data_key is None, the
            entire entry is returned.
        """


        def _pop(data_store, key, data_key=None):
            """ Helper function """
            try:
                entry = data_store[key]
            except KeyError as e:
                raise KeyError("'{}' not in barrier. "
                    "Available keys '{}'".format(key, data_store.keys()))

            # Remove and return entire entry from data store
            # if no specific data_key is provided
            if data_key is None:
                del data_store[key]
                return entry

            # Otherwise return the data entry associated
            # with data key, deleting the entry if it
            # becomes empty
            try:
                data_entry = entry.pop(data_key)

                if len(entry) == 0:
                    del data_store[key]

                return data_entry
            except KeyError as e:
                raise KeyError("Couldn't pop data key '{}' "
                                "in '{}' entry ".format(data_key, key))

        should_wait = not isinstance(timeout, bool)

        # Guard access with lock
        with self._cond:
            while (should_wait and len(self._data_store) == 0
                               and not self.closed()):
                # Break if the timer expires
                if self._cond.wait(timeout) == False:
                    break

            if len(self._data_store) == 0 and self.closed():
                raise ValueError("Data Barrier is closed '{}' '{}'".format(
                                                    key, data_key))

            return _pop(self._data_store, key, data_key)


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