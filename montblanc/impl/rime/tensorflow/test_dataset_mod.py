import unittest
from pprint import pformat

import cppimport
import six
import numpy as np

dsmod = cppimport.imp("dataset_mod")

class TestDatasetmod(unittest.TestCase):
    def test_uvw_antenna(self):
        na = 17
        ntime = 1

        # For both auto correlations and without them
        for auto_cor in (0, 1):
            # Compute default antenna pairs
            ant1, ant2 = np.triu_indices(na, auto_cor)

            # Create random per-antenna UVW coordinates.
            # zeroing the first antenna
            ant_uvw = np.random.random(size=(ntime,na,3)).astype(np.float64)
            ant_uvw[0,0,:] = 0

            time_chunks = np.array([ant1.size], dtype=ant1.dtype)

            # Compute per-baseline UVW coordinates.
            bl_uvw =  (ant_uvw[:,ant1,:] - ant_uvw[:,ant2,:]).reshape(-1, 3)

            # Now recover the per-antenna and per-baseline UVW coordinates.
            rant_uvw = dsmod.antenna_uvw(bl_uvw, ant1, ant2, time_chunks, na)
            rbl_uvw = rant_uvw[:,ant1,:] - rant_uvw[:,ant2,:]

            if not np.allclose(rbl_uvw, bl_uvw):
                self.fail("Recovered baselines do "
                          "not agree\nant1 %s\nant2 %s" % (
                            pformat(ant1), pformat(ant2)))

            if not np.allclose(rant_uvw, ant_uvw):
                self.fail("Recovered antenna do not agree")


    def test_uvw_antenna_missing_bl(self):
        na = 17
        removed_ants_per_time = ([0, 1, 7], [2,10,15,9], [3, 6, 9, 12])

        # For both auto correlations and without them
        for auto_cor in (0, 1):

            def _create_ant_arrays():
                for remove_ants in removed_ants_per_time:
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

                    valid_ants = list(set(six.moves.range(na)).difference(remove_ants))

                    yield valid_ants, remove_ants, ant1, ant2


            valid_ants, remove_ants, ant1, ant2 = zip(*list(_create_ant_arrays()))

            bl_uvw = []

            # Create per-baseline UVW coordinates for each time chunk
            for t, (va, ra, a1, a2) in enumerate(zip(valid_ants, remove_ants, ant1, ant2)):
                # Create random per-antenna UVW coordinates.
                # zeroing the first valid antenna
                ant_uvw = np.random.random(size=(na,3)).astype(np.float64)
                ant_uvw[va[0],:] = 0
                # Create per-baseline UVW coordinates for this time chunk
                bl_uvw.append(ant_uvw[a1,:] - ant_uvw[a2,:])

            # Produced concatenated antenna and baseline uvw arrays
            time_chunks = np.array([a.size for a in ant1], dtype=ant1[0].dtype)
            cant1 = np.concatenate(ant1)
            cant2 = np.concatenate(ant2)
            cbl_uvw = np.concatenate(bl_uvw)

            # Now recover the per-antenna and per-baseline UVW coordinates
            # for the ntime chunks
            rant_uvw = dsmod.antenna_uvw(cbl_uvw, cant1, cant2, time_chunks, na)

            # Reconstruct the baseline UVW coordinates for each chunk
            rbl_uvw = np.concatenate([rant_uvw[t,a1,:] - rant_uvw[t,a2,:]
                        for t, (a1, a2) in enumerate(zip(ant1, ant2))])

            # Check that they agree
            if not np.allclose(cbl_uvw, rbl_uvw):
                self.fail("Recovered baselines do "
                          "not agree\nant1 %s\nant2 %s" % (
                            pformat(ant1), pformat(ant2)))

            # Check that the coordinates of the removed antenna
            # are nan in each time chunk
            for t, ra in enumerate(remove_ants):
                self.assertTrue(np.all(np.isnan(rant_uvw[t,ra,:])),
                    "Removed antenna '%s' UVW coordinates "
                    "in time chunk '%d' are not nan" % (ra, t))

if __name__ == "__main__":
    unittest.main()