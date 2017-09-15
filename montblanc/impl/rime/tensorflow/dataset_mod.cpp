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
    auto uvw_ref = uvw.unchecked();
    auto antenna1_ref = antenna1.unchecked();
    auto antenna2_ref = antenna2.unchecked();
    auto antenna_uvw_ref = antenna_uvw.mutable_unchecked();

    IT ant1 = antenna1_ref(start);
    IT ant2 = antenna2_ref(start);

    // If ant1 associated with starting row is nan
    // initial values have not yet been assigned. Do so.
    if(std::isnan(antenna_uvw_ref(chunk,ant1,0)))
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
            // We 've already computed antenna coordinates
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
    py::array_t<IT, flags> time_chunks,
    IT nr_of_antenna)
{
    py::gil_scoped_release release;

    if(antenna1.ndim() != 1)
        { throw std::invalid_argument("antenna1 shape should be (nrow,)");}

    if(antenna2.ndim() != 1)
        { throw std::invalid_argument("antenna2 shape should be (nrow,)");}

    if(uvw.ndim() != 2 || uvw.shape(1) != 3)
        { throw std::invalid_argument("uvw shape should be (nrow, 3)");}

    if(nr_of_antenna < 1)
        { throw std::invalid_argument("nr_of_antenna < 1"); }

    IT ntime = time_chunks.size();

    // Create numpy array holding the antenna coordinates
    py::array_t<FT, flags> antenna_uvw({int(ntime), int(nr_of_antenna), 3});

    // nan everything in the array
    for(IT i=0; i< antenna_uvw.size(); ++i)
        { antenna_uvw.mutable_data()[i] = std::numeric_limits<FT>::quiet_NaN(); }

    // Find antenna UVW coordinates for each time chunk
    for(IT t=0, start=0; t<ntime; start += *time_chunks.data(t), ++t)
    {
        IT length = *time_chunks.data(t);

        // Loop twice
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, t, start, start+length);
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, t, start, start+length);
    }

    return antenna_uvw;
}

PYBIND11_MODULE(dataset_mod, m) {
    m.doc() = "auto-compiled c++ extension";

    m.def("antenna_uvw", &antenna_uvw<float, std::int32_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<float, std::int32_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<double, std::int64_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<double, std::int64_t>, py::return_value_policy::move);
}