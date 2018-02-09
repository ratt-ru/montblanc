from future_builtins import zip

from itertools import islice
import math
import numpy as np
import numba

# Coordinate indexing constants
u, v, w = range(3)

try:
    isclose = math.isclose
except AttributeError:
    @numba.jit(nopython=True, nogil=True, cache=True)
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

@numba.jit(nopython=True, nogil=True, cache=True)
def _antenna_uvw_loop(uvw, antenna1, antenna2, ant_uvw,
                                chunk_index, start, end):

    c = chunk_index
    a1 = antenna1[start]
    a2 = antenna2[start]

    # Handle first row separately
    # If a1 associated with starting row is nan
    # initial values have not yet been assigned. Do so.
    if np.isnan(ant_uvw[c,a1,u]):
        ant_uvw[c,a1,u] = 0.0
        ant_uvw[c,a1,v] = 0.0
        ant_uvw[c,a1,w] = 0.0

        # If this is not an auto-correlation
        # assign a2 to inverse of baseline UVW,
        if a1 != a2:
            ant_uvw[c,a2,u] = uvw[start,u]
            ant_uvw[c,a2,v] = uvw[start,v]
            ant_uvw[c,a2,w] = uvw[start,w]

    # Now do the rest of the rows in this chunk
    for row in range(start+1, end):
        a1 = antenna1[row]
        a2 = antenna2[row]

        # Have their coordinates been discovered yet?
        ant1_found = not np.isnan(ant_uvw[c,a1,u])
        ant2_found = not np.isnan(ant_uvw[c,a2,u])

        if ant1_found and ant2_found:
            # We've already computed antenna coordinates
            # for this baseline, ignore it
            pass
        elif not ant1_found and not ant2_found:
            # We can't infer one antenna's coordinate from another
            # Hopefully this can be filled in during another run
            # of this function
            pass
        elif ant1_found and not ant2_found:
            # Infer antenna2's coordinate from antenna1
            #    u12 = u2 - u1 => u2 = u12 + u1
            ant_uvw[c,a2,u] = ant_uvw[c,a1,u] + uvw[row,u]
            ant_uvw[c,a2,v] = ant_uvw[c,a1,v] + uvw[row,v]
            ant_uvw[c,a2,w] = ant_uvw[c,a1,w] + uvw[row,w]
        elif not ant1_found and ant2_found:
            # Infer antenna1's coordinate from antenna2
            #    u12 = u2 - u1 => u1 = u2 - u12
            ant_uvw[c,a1,u] = ant_uvw[c,a2,u] - uvw[row,u]
            ant_uvw[c,a1,v] = ant_uvw[c,a2,v] - uvw[row,v]
            ant_uvw[c,a1,w] = ant_uvw[c,a2,w] - uvw[row,w]
        else:
            raise ValueError("Illegal Condition")


@numba.jit(nopython=True, nogil=True, cache=True)
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

        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, ci, start, end)
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

        ant1_uvw = ant_uvw[ci,ant1,:]
        ant2_uvw = ant_uvw[ci,ant2,:]
        ruvw = ant2_uvw - ant1_uvw

        # Identifty rows where any of the UVW components differed
        close = np.isclose(ruvw, cuvw)
        problems = np.nonzero(np.logical_or.reduce(np.invert(close), axis=1))

        for row in problems[0]:
            problem_str.append("[row %d (chunk %d)]: "
                              "original %s "
                              "recovered %s "
                              "ant1 %s "
                              "ant2 %s" % (start+row, ci,
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
        problem_str.append("[chunk %d antenna %d]" % (c,a))

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
