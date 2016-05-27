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
import unittest
import numpy as np
import tempfile
import time

import montblanc
import montblanc.src_types as mbs
import montblanc.util as mbu

class TestSourceUtils(unittest.TestCase):
    """
    TestSourceUtils class defining unit tests for
    montblanc's montblanc.src_type module
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)
        montblanc.setup_test_logging()

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_source_range(self):
        """
        Test that, given a source range, the returned dictionary
        contains the source types within that range
        """
        src_dict = mbs.sources_to_nr_vars({'point':10, 'gaussian':20, 'sersic':40})

        D = mbs.source_range(0, 45, src_dict)
        self.assertTrue(D['npsrc'] == 10 and D['ngsrc'] == 20 and D['nssrc'] == 15)

        D = mbs.source_range(3, 45, src_dict)
        self.assertTrue(D['npsrc'] == 7 and D['ngsrc'] == 20 and D['nssrc'] == 15)

        D = mbs.source_range(13, 45, src_dict)
        self.assertTrue(D['npsrc'] == 0 and D['ngsrc'] == 17 and D['nssrc'] == 15)

        D = mbs.source_range(0, 3, src_dict)
        self.assertTrue(D['npsrc'] == 3 and D['ngsrc'] == 0 and D['nssrc'] == 0)

        D = mbs.source_range(15, 18, src_dict)
        self.assertTrue(D['ngsrc'] == 3)

        D = mbs.source_range(15, 70, src_dict)
        self.assertTrue(D['npsrc'] == 0 and D['ngsrc'] == 15 and D['nssrc'] == 40)

    def test_source_range_tuple(self):
        """
        Test that, given a source range, the returned dictionary
        contains the source types within that range
        """
        src_dict = mbs.sources_to_nr_vars({'point':10, 'gaussian':20, 'sersic':40})

        D = mbs.source_range_tuple(0, 45, src_dict)
        self.assertTrue(D['npsrc'] == (0, 10) and
            D['ngsrc'] == (0, 20) and
            D['nssrc'] == (0, 15))

        D = mbs.source_range_tuple(3, 45, src_dict)
        self.assertTrue(D['npsrc'] == (3, 10) and
            D['ngsrc'] == (0, 20) and
            D['nssrc'] == (0, 15))

        D = mbs.source_range_tuple(13, 45, src_dict)
        self.assertTrue(D['npsrc'] == (0, 0) and
            D['ngsrc'] == (3, 20) and
            D['nssrc'] == (0, 15))

        D = mbs.source_range_tuple(0, 3, src_dict)
        self.assertTrue(D['npsrc'] == (0, 3) and
            D['ngsrc'] == (0, 0) and
            D['nssrc'] == (0, 0))

        D = mbs.source_range_tuple(15, 18, src_dict)
        self.assertTrue(D['npsrc'] == (0, 0) and
            D['ngsrc'] == (5, 8) and
            D['nssrc'] == (0, 0))

        D = mbs.source_range_tuple(15, 70, src_dict)
        self.assertTrue(D['npsrc'] == (0, 0) and
            D['ngsrc'] == (5, 20) and
            D['nssrc'] == (0, 40))

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSourceUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)
