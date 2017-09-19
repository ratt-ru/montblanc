/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-fvisibility=hidden']
%>
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

constexpr unsigned int flags = py::array::c_style;

// More intuitive indexing
enum { u=0, v=1, w=2 };

template <typename FT, typename IT>
void _antenna_uvw_loop(
    py::array_t<FT, flags> & uvw,
    py::array_t<IT, flags> & antenna1,
    py::array_t<IT, flags> & antenna2,
    py::array_t<FT, flags> & antenna_uvw,
    IT chunk, IT start, IT end)
{
    // Do unchecked bounds access of array data
    auto uvw_ref = uvw.unchecked();
    auto antenna1_ref = antenna1.unchecked();
    auto antenna2_ref = antenna2.unchecked();
    auto antenna_uvw_ref = antenna_uvw.mutable_unchecked();

    IT ant1 = antenna1_ref(start);
    IT ant2 = antenna2_ref(start);

    // If ant1 associated with starting row is nan
    // initial values have not yet been assigned. Do so.
    if(std::isnan(antenna_uvw_ref(chunk,ant1,u)))
    {
        // Choose first antenna value as the origin
        antenna_uvw_ref(chunk,ant1,u) = 0.0;
        antenna_uvw_ref(chunk,ant1,v) = 0.0;
        antenna_uvw_ref(chunk,ant1,w) = 0.0;

        // Only set the second antenna value if
        // this is not an auto-correlation.
        if(ant1 != ant2)
        {
            antenna_uvw_ref(chunk,ant2,u) = -uvw_ref(start,u);
            antenna_uvw_ref(chunk,ant2,v) = -uvw_ref(start,v);
            antenna_uvw_ref(chunk,ant2,w) = -uvw_ref(start,w);
        }
    }

    // Handle the rest of the rows
    for(IT row=start+1; row < end; ++row)
    {
        IT ant1 = antenna1_ref(row);
        IT ant2 = antenna2_ref(row);

        // Reference each antenna's possibly discovered
        // UVW coordinate in the array
        FT * ant1_uvw = antenna_uvw_ref.mutable_data(chunk, ant1);
        FT * ant2_uvw = antenna_uvw_ref.mutable_data(chunk, ant2);

        // Are antenna one and two's u coordinate nan
        // and therefore is the coordinate discovered?
        bool ant1_found = !std::isnan(ant1_uvw[u]);
        bool ant2_found = !std::isnan(ant2_uvw[u]);

        if(ant1_found && ant2_found)
        {
            // We've already computed antenna coordinates
            // for this baseline, ignore it
        }
        else if(!ant1_found && !ant2_found)
        {
            // We can't infer one antenna's coordinate from another
            // Hopefully this can be filled in during another run
            // of this function
        }
        else if(ant1_found && !ant2_found)
        {
            // Infer antenna2's coordinate from antenna1
            //    u12 = u1 - u2
            // => u2 = u1 - u12
            ant2_uvw[u] = ant1_uvw[u] - uvw_ref(row,u);
            ant2_uvw[v] = ant1_uvw[v] - uvw_ref(row,v);
            ant2_uvw[w] = ant1_uvw[w] - uvw_ref(row,w);
        }
        else if (!ant1_found && ant2_found)
        {
            // Infer antenna1's coordinate from antenna2
            //    u12 = u1 - u2
            // => u1 = u12 + u2
            ant1_uvw[u] = uvw_ref(row,u) + ant2_uvw[u];
            ant1_uvw[v] = uvw_ref(row,v) + ant2_uvw[v];
            ant1_uvw[w] = uvw_ref(row,w) + ant2_uvw[w];
        }
    }
}

template <typename FT, typename IT>
py::array_t<FT, flags> antenna_uvw(
    py::array_t<FT, flags> uvw,
    py::array_t<IT, flags> antenna1,
    py::array_t<IT, flags> antenna2,
    py::array_t<IT, flags> chunks,
    py::kwargs kwargs)
{
    if(!kwargs.contains("nr_of_antenna"))
        { throw std::invalid_argument("antenna_uvw keyword argument"
                                    "'nr_of_antenna' not set"); }

    int nr_of_antenna = kwargs["nr_of_antenna"].cast<IT>();

    // Drop the GIL
    py::gil_scoped_release release;

    // Do some shape checking
    int nr_of_uvw = uvw.shape(1);

    if(antenna1.ndim() != 1)
        { throw std::invalid_argument("antenna1 shape should be (nrow,)");}

    if(antenna2.ndim() != 1)
        { throw std::invalid_argument("antenna2 shape should be (nrow,)");}

    if(uvw.ndim() != 2 || nr_of_uvw != 3)
        { throw std::invalid_argument("uvw shape should be (nrow, 3)");}

    if(nr_of_antenna < 1)
        { throw std::invalid_argument("nr_of_antenna < 1"); }

    // Create numpy array holding the antenna coordinates
    py::array_t<FT, flags> antenna_uvw({int(chunks.size()), int(nr_of_antenna), nr_of_uvw});

    auto chunks_ref = chunks.unchecked();

    // nan everything in the array
    for(IT i=0; i< antenna_uvw.size(); ++i)
        { antenna_uvw.mutable_data()[i] = std::numeric_limits<FT>::quiet_NaN(); }

    // Find antenna UVW coordinates for each chunk
    for(IT c=0, start=0; c<chunks.size(); start += chunks_ref(c), ++c)
    {
        // Loop twice
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw,
                            c, start, start+chunks_ref(c));
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw,
                            c, start, start+chunks_ref(c));
    }

    return antenna_uvw;
}

auto constexpr antenna_uvw_docstring = R"doc(
    Computes per-antenna UVW coordinates from baseline `uvw`,
    `antenna1` and `antenna2` coordinates logically grouped
    into chunks of baselines per unique timestep.

    The example below illustrates two baseline groupings
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
    the baseline indices need not be properly ordered.

    If it is not possible to derive coordinates for an antenna, it's coordinate
    will be set to nan.

    Notes
    -----
    The indexing and chunking arrays must use the same integral types:
    :code:`np.int32` or :code:`np.int64`.

    Parameters
    ----------
    uvw : np.ndarray
        Baseline UVW coordinates of shape (row, 3)
    antenna1 : np.ndarray
        Baseline first antenna of shape (row,)
    antenna2 : np.ndarray
        Baseline second antenna of shape (row,)
    chunks : np.ndarray
        Number of baselines per unique timestep with shape (utime,)
        :code:`np.sum(chunks) == row` should hold.
    nr_of_antenna : int
        Total number of antenna in the solution.

    Returns
    -------
    np.ndarray
        Antenna UVW coordinates of shape (utime, nr_of_antenna, 3)

)doc";

PYBIND11_MODULE(dataset_mod, m) {
    m.doc() = "auto-compiled c++ extension";

    m.def("antenna_uvw", &antenna_uvw<float, std::int32_t>,
        py::return_value_policy::move, antenna_uvw_docstring);
    m.def("antenna_uvw", &antenna_uvw<float, std::int64_t>,
        py::return_value_policy::move, antenna_uvw_docstring);
    m.def("antenna_uvw", &antenna_uvw<double, std::int32_t>,
        py::return_value_policy::move, antenna_uvw_docstring);
    m.def("antenna_uvw", &antenna_uvw<double, std::int64_t>,
        py::return_value_policy::move, antenna_uvw_docstring);
}