import numpy as np


def _uniq_log2_range(start, size, div):
    """
    Produce unique integers in the start, start+size range
    with a log2 distribution
    """
    start = np.log2(start)
    size = np.log2(size)
    int_values = np.int32(np.logspace(start, size, div, base=2)[:-1])

    return np.flipud(np.unique(int_values))


def row_time_reduction(time_chunks, row_chunks):
    yield [('source', 50)]

    ntime = sum(time_chunks)
    time_counts = _uniq_log2_range(1, ntime, 50)

    for time_count in time_counts:
        rows = sum(row_chunks[:time_count])
        times = sum(time_chunks[:time_count])

        yield [('row', rows), ('time', times)]


def budget(schemas, dims, mem_budget, reduce_fn):
    """
    Reduce dimension values in `dims` according to
    strategy specified in generator `reduce_fn`
    until arrays in `schemas` fit within specified `mem_budget`.

    Parameters
    ----------
    schemas : dict or sequence of dict
        Dictionary of array schemas, of the form
        :code:`{name : {"dtype": dtype, "dims": (d1,d2,...,dn)}}`
    dims : dict
        Dimension size mapping, of the form
        :code:`{"d1": i, "d2": j, ..., "dn": k}
    mem_budget : int
        Number of bytes defining the memory budget
    reduce_fn : callable
        Generator yielding a lists of dimension reduction tuples.
        For example:

        .. code-block:: python

            def red_gen():
                yield [('source', 50)]
                yield [('time', 100), ('row', 10000)]
                yield [('time', 50), ('row', 1000)]
                yield [('time', 20), ('row', 100)]

    Returns
    -------
    dict
        A :code:`{dim: size}` mapping of
        dimension reductions that fit the
        schema within the memory budget.
    """

    # Promote to list
    if not isinstance(schemas, (tuple, list)):
        schemas = [schemas]

    array_details = {n: (a['dims'], np.dtype(a['dtype']))
                     for schema in schemas
                     for n, a in schema.items()}

    applied_reductions = {}

    def get_bytes(dims, arrays):
        """ Get number of bytes in the schema """
        return sum(np.product(tuple(dims[d] for d in a[0]))*a[1].itemsize
                   for a in arrays.values())

    bytes_required = get_bytes(dims, array_details)

    for reduction in reduce_fn():
        if bytes_required > mem_budget:
            for dim, size in reduction:
                dims[dim] = size
                applied_reductions[dim] = size

            bytes_required = get_bytes(dims, array_details)
        else:
            break

    return bytes_required, applied_reductions
