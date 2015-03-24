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
import montblanc.factory
import montblanc.util as mbu


class TestUtils(unittest.TestCase):
    """
    TestUtil class defining unit tests for
    montblanc's montblanc.util module
    """

    def setUp(self):
        """ Set up each test case """
        np.random.seed(int(time.time()) & 0xFFFFFFFF)

        # Add a handler that outputs INFO level logging
        fh = logging.FileHandler('test.log')
        fh.setLevel(logging.INFO)

        montblanc.log.addHandler(fh)
        montblanc.log.setLevel(logging.INFO)

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_numeric_shapes(self):
        """ Test that we can convert shapes with string arguments into numeric shapes """

        shape_one = (5,'ntime','nchan')
        shape_two = (10, 'nsrc')
        ignore = ['ntime']

        ri = np.random.randint

        gns = mbu.shape_from_str_tuple
        P = { 'ntime' : ri(5,10), 'nbl' : ri(3,5), 'nchan' : 4,
            'nsrc' : ri(4,10) }

        ntime, nbl, nchan, nsrc = P['ntime'], P['nbl'], P['nchan'], P['nsrc']

        self.assertTrue(gns(shape_one, P) == (5,ntime,nchan))
        self.assertTrue(gns(shape_two, P) == (10,nsrc))

        self.assertTrue(gns(shape_one, P, ignore=ignore) == (5,nchan))
        self.assertTrue(gns(shape_two, P, ignore=ignore) == (10,nsrc))

    def test_array_conversion(self):
        """
        Test that we can produce NumPy code that automagically translates
        between shapes with string expressions
        """

        props = { 'ntime' : np.random.randint(5,10),
                    'nbl' : np.random.randint(3,5),
                    'nchan' : 16 }

        ntime, nbl, nchan = props['ntime'], props['nbl'], props['nchan']

        f = mbu.array_convert_function(
            (3,'ntime*nchan','nbl'), ('nchan', 'nbl*ntime*3'), props)

        ary = np.random.random(size=(3, ntime*nchan, nbl))
        self.assertTrue(np.all(f(ary) ==
            ary.reshape(3,ntime,nchan,nbl) \
            .transpose(2,3,1,0) \
            .reshape(nchan, nbl*ntime*3)))

    def test_eval_expr(self):
        """ Test evaluation expression and parsing """
        props = { 'ntime' : 7, 'nbl' : 4 }

        self.assertTrue(mbu.eval_expr(
            '1+2*ntime+nbl', props) == (1+2*props['ntime']+props['nbl']))

        self.assertTrue(mbu.eval_expr(
            'ntime*nbl*3', props) == props['ntime']*props['nbl']*3)

        self.assertTrue(mbu.eval_expr_names_and_nrs(
            '1+2*ntime+nbl') == [1,2,'ntime','nbl'])

        self.assertTrue(mbu.eval_expr_names_and_nrs(
            'ntime*nbl+3-1') == ['ntime','nbl',3,1])

        self.assertTrue(mbu.eval_expr_names_and_nrs(
            'ntime*3+1-nbl') == ['ntime',3,1,'nbl'])

    def test_baseline_antenna_nrs(self):
        """ Test conversion between antenna and baseline numbers """

        def do_check(na, nbl, nbl_auto):
            # get nr baselines from nr of antenna
            self.assertTrue(mbu.nr_of_baselines(na) == nbl)
            self.assertTrue(mbu.nr_of_baselines(na, False) == nbl)
            self.assertTrue(mbu.nr_of_baselines(na, True) == nbl_auto)

            # get nr antenna from nr of baselines
            self.assertTrue(mbu.nr_of_antenna(nbl) == na)
            self.assertTrue(mbu.nr_of_antenna(nbl, False) == na)
            self.assertTrue(mbu.nr_of_antenna(nbl_auto, True) == na)

        do_check(7, 7*6//2, 7*8//2)                                 # KAT7
        do_check(14, 14*13//2, 14*15//2)                       # Westerbork
        do_check(27, 27*26//2, 27*28//2)                       # VLA
        do_check(64, 64*63//2, 64*65//2)                       # MeerKAT
        do_check(3500, 3500*3499//2, 3500*3501//2)   # SKA

    def test_sky_model(self):
        """ Test sky model file loading """

        sky_model_file_contents = (
            '# format npsrc: l m I Q U V\n'
            '11, 12, 13, 14, 15, 16\n'
            '21, 22, 23, 24, 25, 26\n'
            '# format ngsrc: l m I Q U V el em eR\n'
            '31, 32, 33, 34, 35, 36, 37, 38, 39\n'
            '41, 42, 43, 44, 45, 46, 47, 48, 49\n')

        with tempfile.NamedTemporaryFile('w') as f:
            f.write(sky_model_file_contents)
            f.flush()

            result = mbu.parse_sky_model(f.name)
            A, S = result.arrays, result.src_counts

        self.assertTrue(S['npsrc'] == 2)
        self.assertTrue(S['ngsrc'] == 2)

        self.assertTrue(A['l'] == ['11', '21', '31', '41'])
        self.assertTrue(A['m'] == ['12', '22', '32', '42'])
        self.assertTrue(A['I'] == ['13', '23', '33', '43'])
        self.assertTrue(A['Q'] == ['14', '24', '34', '44'])
        self.assertTrue(A['U'] == ['15', '25', '35', '45'])
        self.assertTrue(A['V'] == ['16', '26', '36', '46'])
        self.assertTrue(A['el'] == ['37', '47'])
        self.assertTrue(A['em'] == ['38', '48'])
        self.assertTrue(A['eR'] == ['39', '49'])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(suite)

