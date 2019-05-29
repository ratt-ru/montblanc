import unittest
from pprint import pformat

from six.moves import range
import numpy as np

from montblanc.util import antenna_uvw


class TestAntennaUvWDecomposition(unittest.TestCase):
    def test_uvw_antenna(self):
        na = 17
        ntime = 1

        # For both auto correlations and without them
        for auto_cor in (0, 1):
            # Compute default antenna pairs
            ant1, ant2 = np.triu_indices(na, auto_cor)

            # Create random per-antenna UVW coordinates.
            # zeroing the first antenna
            ant_uvw = np.random.random(size=(ntime, na, 3)).astype(np.float64)
            ant_uvw[0, 0, :] = 0

            time_chunks = np.array([ant1.size], dtype=ant1.dtype)

            # Compute per-baseline UVW coordinates.
            bl_uvw = (ant_uvw[:, ant1, :] - ant_uvw[:, ant2, :]).reshape(-1, 3)

            # Now recover the per-antenna and per-baseline UVW coordinates.
            rant_uvw = antenna_uvw(bl_uvw, ant1, ant2, time_chunks,
                                   nr_of_antenna=na, check_decomposition=True)

    def test_uvw_disjoint(self):

        # Three initially disjoint baselines here, but the last baseline [2, 9]
        # connects the first and the last
        # Set 1: 0, 1, 2, 3
        # Set 2: 4, 5, 6, 7, 8
        # Set 3: 8, 10, 11, 12
        # Connection between Set 1 and Set 3 is the last baseline [2, 9]
        ant1 = np.array([1,  2,  3,  4,  5,  5,  7,  9, 10, 11,  2])
        ant2 = np.array([2,  2,  0,  5,  5,  6,  8, 10, 11, 12,  9])

        na = np.unique(np.concatenate([ant1, ant2])).size
        ntime = 1

        # Create random per-antenna UVW coordinates.
        # zeroing the first antenna
        ant_uvw = np.random.random(size=(ntime, na, 3)).astype(np.float64)
        ant_uvw[0, 0, :] = 0

        time_chunks = np.array([ant1.size], dtype=ant1.dtype)

        # Compute per-baseline UVW coordinates.
        bl_uvw = (ant_uvw[:, ant1, :] - ant_uvw[:, ant2, :]).reshape(-1, 3)

        # Now recover the per-antenna and per-baseline UVW coordinates.
        rant_uvw = antenna_uvw(bl_uvw, ant1, ant2, time_chunks,
                               nr_of_antenna=na, check_decomposition=True)

    def test_uvw_antenna_missing_bl_impl(self):
        na = 17
        removed_ants_per_time = ([0, 1, 7], [2, 10, 15, 9], [3, 6, 9, 12])

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

                    # Remove any baselines containing flagged antenna
                    reduce_tuple = tuple(a != ra for a in (ant1, ant2)
                                         for ra in remove_ants)

                    keep = np.logical_and.reduce(reduce_tuple)
                    ant1 = ant1[keep]
                    ant2 = ant2[keep]

                    valid_ants = list(set(range(na)).difference(remove_ants))

                    yield valid_ants, remove_ants, ant1, ant2

            tup = zip(*list(_create_ant_arrays()))
            valid_ants, remove_ants, ant1, ant2 = tup

            bl_uvw = []

            # Create per-baseline UVW coordinates for each time chunk
            it = enumerate(zip(valid_ants, remove_ants, ant1, ant2))
            for t, (va, ra, a1, a2) in it:
                # Create random per-antenna UVW coordinates.
                # zeroing the first valid antenna
                ant_uvw = np.random.random(size=(na, 3)).astype(np.float64)
                ant_uvw[va[0], :] = 0
                # Create per-baseline UVW coordinates for this time chunk
                bl_uvw.append(ant_uvw[a1, :] - ant_uvw[a2, :])

            # Produced concatenated antenna and baseline uvw arrays
            time_chunks = np.array([a.size for a in ant1], dtype=ant1[0].dtype)
            cant1 = np.concatenate(ant1)
            cant2 = np.concatenate(ant2)
            cbl_uvw = np.concatenate(bl_uvw)

            # Now recover the per-antenna and per-baseline UVW coordinates
            # for the ntime chunks
            rant_uvw = antenna_uvw(cbl_uvw, cant1, cant2, time_chunks,
                                   nr_of_antenna=na, check_decomposition=True)


if __name__ == "__main__":
    unittest.main()
