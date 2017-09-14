import unittest
from pprint import pformat

import cppimport
import six
import numpy as np

dsmod = cppimport.imp("dataset_mod")

class TestDatasetmod(unittest.TestCase):
    def test_uvw_antenna(self):
        na = 17

        # For both auto correlations and without them
        for auto_cor in (0, 1):
            # Compute default antenna pairs
            ant1, ant2 = np.triu_indices(na, auto_cor)

            # Get the unique antenna indices
            ant_i = np.unique(np.concatenate([ant1, ant2]))

            # Create random per-antenna UVW coordinates.
            # zeroing the first antenna
            ant_uvw = np.random.random(size=(na,3)).astype(np.float64)
            ant_uvw[0,:] = 0

            # Compute per-baseline UVW coordinates.
            bl_uvw =  ant_uvw[ant1] - ant_uvw[ant2]

            # Now recover the per-antenna and per-baseline UVW coordinates.
            rant_uvw = dsmod.antenna_uvw(bl_uvw, ant1, ant2)
            rbl_uvw = rant_uvw[ant1] - rant_uvw[ant2]

            if not np.allclose(rbl_uvw, bl_uvw):
                self.fail("Recovered baselines do "
                          "not agree\nant1 %s\nant2 %s" % (
                            pformat(ant1), pformat(ant2)))

            if not np.allclose(rant_uvw, ant_uvw):
                self.fail("Recovered antenna do not agree")


    def test_uvw_antenna_missing_bl(self):
        na = 17
        remove_ants = [0, 1, 7]
        valid_ants = list(set(six.moves.range(na)).difference(remove_ants))

        # For both auto correlations and without them
        for auto_cor in (0, 1):
            # Compute default antenna pairs
            ant1, ant2 = np.triu_indices(na, auto_cor)

            # Shuffle the antenna indices
            idx = np.arange(ant1.size)
            np.random.shuffle(idx)

            ant1 = ant1[idx]
            ant2 = ant2[idx]

            # Remove any baselines containing flagged antennae
            reduce_tuple = tuple(a != ra for a in (ant1, ant2)
                                        for ra in remove_ants)

            keep = np.logical_and.reduce(reduce_tuple)
            ant1 = ant1[keep]
            ant2 = ant2[keep]

            # Get the unique antenna indices, and from
            # this, the maximum possible number of antenna
            ant_i = np.unique(np.concatenate([ant1, ant2]))
            na = np.max(ant_i)+1

            # Create random per-antenna UVW coordinates.
            # zeroing the first antenna
            ant_uvw = np.random.random(size=(na,3)).astype(np.float64)
            ant_uvw[valid_ants[0],:] = 0

            # Compute per-baseline UVW coordinates.
            bl_uvw =  ant_uvw[ant1] - ant_uvw[ant2]

            # Now recover the per-antenna and per-baseline UVW coordinates.
            rant_uvw = dsmod.antenna_uvw(bl_uvw, ant1, ant2)

            rbl_uvw = rant_uvw[ant1] - rant_uvw[ant2]

            if not np.allclose(bl_uvw, rbl_uvw):
                self.fail("Recovered baselines do "
                          "not agree\nant1 %s\nant2 %s" % (
                            pformat(ant1), pformat(ant2)))

            # All missing antenna's are nanned
            self.assertTrue(np.all(np.isnan(rant_uvw[remove_ants])))

if __name__ == "__main__":
    unittest.main()