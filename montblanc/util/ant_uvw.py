try:
    from future_builtins import zip
except:
    pass

from itertools import islice
import math
import numpy as np
from numba import jit, generated_jit

# Coordinate indexing constants
u, v, w = list(range(3))

try:
    isclose = math.isclose
except AttributeError:
    @jit(nopython=True, nogil=True, cache=True)
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@jit(nopython=True, nogil=True, cache=True)
def _antenna_uvw_loop(uvw, antenna1, antenna2, ant_uvw,
                      chunk_index, start, end):

    c = chunk_index

    # Cluster (first antenna) associated with each antenna
    clusters = np.full((ant_uvw.shape[1],), -1, dtype=antenna1.dtype)

    # Iterate over rows in chunk
    for row in range(start, end):
        a1 = antenna1[row]
        a2 = antenna2[row]

        # Have they been clustered yet?
        cl1 = clusters[a1]
        cl2 = clusters[a2]

        # Both new -- start a new cluster relative to a1
        if cl1 == -1 and cl2 == -1:
            clusters[a1] = clusters[a2] = a1
            ant_uvw[c, a1, u] = 0.0
            ant_uvw[c, a1, v] = 0.0
            ant_uvw[c, a1, w] = 0.0

            # If this is not an auto-correlation
            # assign a2 to inverse of baseline UVW,
            if a1 != a2:
                ant_uvw[c, a2, u] = uvw[row, u]
                ant_uvw[c, a2, v] = uvw[row, v]
                ant_uvw[c, a2, w] = uvw[row, w]

        # if either antenna has not been clustered,
        # infer its coordinate from the clustered one
        elif cl1 != -1 and cl2 == -1:
            clusters[a2] = cl1
            ant_uvw[c, a2, u] = ant_uvw[c, a1, u] + uvw[row, u]
            ant_uvw[c, a2, v] = ant_uvw[c, a1, v] + uvw[row, v]
            ant_uvw[c, a2, w] = ant_uvw[c, a1, w] + uvw[row, w]
        elif cl1 == -1 and cl2 != -1:
            clusters[a1] = cl2
            ant_uvw[c, a1, u] = ant_uvw[c, a2, u] - uvw[row, u]
            ant_uvw[c, a1, v] = ant_uvw[c, a2, v] - uvw[row, v]
            ant_uvw[c, a1, w] = ant_uvw[c, a2, w] - uvw[row, w]
        # Both clustered. If clusters differ, merge them
        elif cl1 != -1 and cl2 != -1:
            if cl1 != cl2:
                # how much do we need to add to the current cluster2
                # reference position to make the baseline a2 - a1 consistent?
                u_off = uvw[row, u] - (ant_uvw[c, a2, u] - ant_uvw[c, a1, u])
                v_off = uvw[row, v] - (ant_uvw[c, a2, v] - ant_uvw[c, a1, v])
                w_off = uvw[row, w] - (ant_uvw[c, a2, w] - ant_uvw[c, a1, w])

                # Merge cluster2 into cluster1
                for ant, cluster in enumerate(clusters):
                    if cluster == cl2:
                        clusters[ant] = cl1
                        ant_uvw[c, ant, u] += u_off
                        ant_uvw[c, ant, v] += v_off
                        ant_uvw[c, ant, w] += w_off

        # Shouldn't ever occur
        else:
            raise ValueError("Illegal Condition")


@jit(nopython=True, nogil=True, cache=True)
def _antenna_uvw(uvw, antenna1, antenna2, chunks, nr_of_antenna):
    """ numba implementation of antenna_uvw """

    if antenna1.ndim != 1:
        raise ValueError("antenna1 shape should be (row,)")

    if antenna2.ndim != 1:
        raise ValueError("antenna2 shape should be (row,)")

    if uvw.ndim != 2 or uvw.shape[1] != 3:
        raise ValueError("uvw shape should be (row, 3)")

    if not (uvw.shape[0] == antenna1.shape[0] == antenna2.shape[0]):
        raise ValueError("First dimension of uvw, antenna1 "
                         "and antenna2 do not match")

    if chunks.ndim != 1:
        raise ValueError("chunks shape should be (utime,)")

    if nr_of_antenna < 1:
        raise ValueError("nr_of_antenna < 1")

    ant_uvw_shape = (chunks.shape[0], nr_of_antenna, 3)
    antenna_uvw = np.full(ant_uvw_shape, np.nan, dtype=uvw.dtype)

    start = 0

    for ci, chunk in enumerate(chunks):
        end = start + chunk

        # one pass should be enough!
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, ci, start, end)

        start = end

    return antenna_uvw


class AntennaUVWDecompositionError(Exception):
    pass


def _raise_decomposition_errors(uvw, antenna1, antenna2,
                                chunks, ant_uvw, max_err):
    """ Raises informative exception for an invalid decomposition """

    start = 0

    problem_str = []

    for ci, chunk in enumerate(chunks):
        end = start + chunk

        ant1 = antenna1[start:end]
        ant2 = antenna2[start:end]
        cuvw = uvw[start:end]

        ant1_uvw = ant_uvw[ci, ant1, :]
        ant2_uvw = ant_uvw[ci, ant2, :]
        ruvw = ant2_uvw - ant1_uvw

        # Identifty rows where any of the UVW components differed
        close = np.isclose(ruvw, cuvw)
        problems = np.nonzero(np.logical_or.reduce(np.invert(close), axis=1))

        for row in problems[0]:
            problem_str.append("[row %d [%d, %d] (chunk %d)]: "
                               "original %s recovered %s "
                               "ant1 %s ant2 %s" % (
                                    start+row, ant1[row], ant2[row], ci,
                                    cuvw[row], ruvw[row],
                                    ant1_uvw[row], ant2_uvw[row]))

            # Exit inner loop early
            if len(problem_str) >= max_err:
                break

        # Exit outer loop early
        if len(problem_str) >= max_err:
            break

        start = end

    # Return early if nothing was wrong
    if len(problem_str) == 0:
        return

    # Add a preamble and raise exception
    problem_str = ["Antenna UVW Decomposition Failed",
                   "The following differences were found "
                   "(first 100):"] + problem_str
    raise AntennaUVWDecompositionError('\n'.join(problem_str))


class AntennaMissingError(Exception):
    pass


def _raise_missing_antenna_errors(ant_uvw, max_err):
    """ Raises an informative error for missing antenna """

    # Find antenna uvw coordinates where any UVW component was nan
    # nan + real == nan
    problems = np.nonzero(np.add.reduce(np.isnan(ant_uvw), axis=2))
    problem_str = []

    for c, a in zip(*problems):
        problem_str.append("[chunk %d antenna %d]" % (c, a))

        # Exit early
        if len(problem_str) >= max_err:
            break

    # Return early if nothing was wrong
    if len(problem_str) == 0:
        return

    # Add a preamble and raise exception
    problem_str = ["Antenna were missing"] + problem_str
    raise AntennaMissingError('\n'.join(problem_str))


def antenna_uvw(uvw, antenna1, antenna2, chunks,
                nr_of_antenna, check_missing=False,
                check_decomposition=False, max_err=100):
    """
    Computes per-antenna UVW coordinates from baseline ``uvw``,
    ``antenna1`` and ``antenna2`` coordinates logically grouped
    into baseline chunks.

    The example below illustrates two baseline chunks
    of size 6 and 5, respectively.

    .. code-block:: python

        uvw = ...
        ant1 = np.array([0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1], dtype=np.int32)
        ant2 = np.array([1, 2, 3, 2, 3, 3, 1, 2, 3, 1, 2], dtype=np.int32)
        chunks = np.array([6, 5], dtype=np.int32)

        ant_uv = antenna_uvw(uvw, ant1, ant2, chunks, nr_of_antenna=4)

    The first antenna of the first baseline of a chunk is chosen as the origin
    of the antenna coordinate system, while the second antenna is set to the
    negative of the baseline UVW coordinate. Subsequent antenna UVW coordinates
    are iteratively derived from the first two coordinates. Thus,
    the baseline indices need not be properly ordered (within the chunk).

    If it is not possible to derive coordinates for an antenna,
    it's coordinate will be set to nan.

    Parameters
    ----------
    uvw : np.ndarray
        Baseline UVW coordinates of shape (row, 3)
    antenna1 : np.ndarray
        Baseline first antenna of shape (row,)
    antenna2 : np.ndarray
        Baseline second antenna of shape (row,)
    chunks : np.ndarray
        Number of baselines per unique timestep with shape (chunks,)
        :code:`np.sum(chunks) == row` should hold.
    nr_of_antenna : int
        Total number of antenna in the solution.
    check_missing (optional) : bool
        If ``True`` raises an exception if it was not possible
        to compute UVW coordinates for all antenna (i.e. some were nan).
        Defaults to ``False``.
    check_decomposition (optional) : bool
        If ``True``, checks that the antenna decomposition accurately
        reproduces the coordinates in ``uvw``, or that
        :code:`ant_uvw[c,ant1,:] - ant_uvw[c,ant2,:] == uvw[s:e,:]`
        where ``s`` and ``e`` are the start and end rows
        of chunk ``c`` respectively. Defaults to ``False``.
    max_err (optional) : integer
        Maximum numbers of errors when checking for missing antenna
        or innacurate decompositions. Defaults to ``100``.

    Returns
    -------
    np.ndarray
        Antenna UVW coordinates of shape (chunks, nr_of_antenna, 3)
    """

    ant_uvw = _antenna_uvw(uvw, antenna1, antenna2, chunks, nr_of_antenna)

    if check_missing:
        _raise_missing_antenna_errors(ant_uvw, max_err=max_err)

    if check_decomposition:
        _raise_decomposition_errors(uvw, antenna1, antenna2, chunks,
                                    ant_uvw, max_err=max_err)

    return ant_uvw
