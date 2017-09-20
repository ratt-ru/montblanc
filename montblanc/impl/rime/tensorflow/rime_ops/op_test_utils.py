import numpy as np

def random_baselines(chunks, nr_of_antenna, auto_correlations=False):
    """
    Generates randomised `uvw`, coordinates, as well as
    `antenna1`, `antenna2` and `time_index` indices,
    for the given list of rows per unique time (`chunks`).

    Parameters
    ----------
    chunks : tuple, list or np.ndarray
        List of rows per unique time. Shape (utime,)
    nr_of_antenna : int
        Number of antenna
    auto_correlations (optional) : {False, True}
        Include auto correlation baselines

    Returns
    -------
    tuple
        Tuple of four np.ndarrays,
        `uvw`, `antenna1`, `antenna2` and `time_index`
        each with shape (sum(chunks),)
    """

    # Promote chunks to numpy array
    if isinstance(chunks, (tuple, list)):
        chunks = np.array(chunks)
    elif isinstance(chunks, int):
        chunks = np.array([chunks])

    # number of unique times and antenna
    utime = chunks.shape[0]
    na = nr_of_antenna

    # Basic antenna combinations
    k = 0 if auto_correlations == True else 1
    ant1, ant2 = map(lambda x: np.int32(x), np.triu_indices(na, k))

    # Create Antenna uvw coordinates, zeroing the first
    ant_uvw = np.random.random(size=(utime, na, 3)).astype(np.float64)
    ant_uvw[:,0,:] = 0

    # Create baseline uvw coordinates
    bl_uvw = ant_uvw[:,ant1] - ant_uvw[:,ant2]
    bl_index = np.arange(ant1.size)

    def _chunk_baselines(ut, chunk_rows):
        """ Returns baslines for a chunk at index `ut` with rows `chunk_rows` """

        # Shuffle canonical baselines and take the first chunk_rows
        index = bl_index.copy()
        np.random.shuffle(index)
        index = index[:chunk_rows]

        return (bl_uvw[ut,index], ant1[index], ant2[index],
                np.full(index.size, ut, dtype=np.int32))

    # Get list of uvw, ant1, ant2 chunks. zip(*(...)) transposes
    uvw, ant1, ant2, tindex = tuple(np.concatenate(a) for a
                                in zip(*(_chunk_baselines(ut, cr)
                                for ut, cr in enumerate(chunks))))

    assert ant1.size == np.sum(chunks)

    return uvw, ant1, ant2, tindex
