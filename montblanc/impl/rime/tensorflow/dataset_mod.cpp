/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-fvisibility=hidden']
%>
*/

#include <algorithm>
#include <cstdint>

#include <limits>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

constexpr unsigned int flags = py::array::c_style;

template <typename FT>
class UVWCoordinate
{
public:
    FT u,v,w;

    UVWCoordinate(const FT & u=FT(),
                const FT & v=FT(),
                const FT & w=FT())
        : u(u), v(v), w(w) {}
};

template <typename FT, typename IT>
using AntennaUVWMap = std::unordered_map<IT, UVWCoordinate<FT>>;

template <typename FT, typename IT>
void _antenna_uvw_loop(
    py::array_t<FT, flags> & uvw,
    py::array_t<IT, flags> & antenna1,
    py::array_t<IT, flags> & antenna2,
    AntennaUVWMap<FT, IT> & antenna_uvw,
    IT start, IT end)
{
    // Special case, infer the first (two) antenna coordinate(s)
    // from the first row
    if(antenna_uvw.size() == 0)
    {
        IT ant1 = *antenna1.data(start);
        IT ant2 = *antenna2.data(start);
        const FT * u = uvw.data(start,0);
        const FT * v = uvw.data(start,1);
        const FT * w = uvw.data(start,2);

        // Choose first antenna value as the origin
        antenna_uvw.insert({ ant1, UVWCoordinate<FT>(0,0,0) });

        // If this is not an auto-correlation
        // set second antenna value as baseline inverse
        if(ant1 != ant2)
        {
            antenna_uvw.insert({ ant2, UVWCoordinate<FT>(-*u, -*v, -*w) });
        }
    }

    // Handle the rest of the rows
    for(IT row=start+1; row < end; ++row)
    {
        IT ant1 = *antenna1.data(row);
        IT ant2 = *antenna2.data(row);
        const FT * u = uvw.data(row,0);
        const FT * v = uvw.data(row,1);
        const FT * w = uvw.data(row,2);

        // Lookup any existing antenna values
        auto ant1_lookup = antenna_uvw.find(ant1);
        auto ant2_lookup = antenna_uvw.find(ant2);

        bool ant1_found = ant1_lookup != antenna_uvw.end();
        bool ant2_found = ant2_lookup != antenna_uvw.end();

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
            const auto & ant1_uvw = ant1_lookup->second;

            antenna_uvw.insert({ ant2, UVWCoordinate<FT>(
                ant1_uvw.u - *u,
                ant1_uvw.v - *v,
                ant1_uvw.w - *w) });
        }
        else if (!ant1_found && ant2_found)
        {
            // Infer antenna1's coordinate from antenna1
            //    u12 = u1 - u2
            // => u1 = u12 + u2

            const auto & ant2_uvw = ant2_lookup->second;

            antenna_uvw.insert({ ant1, UVWCoordinate<FT>(
                *u + ant2_uvw.u,
                *v + ant2_uvw.v,
                *w + ant2_uvw.w) });
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

    AntennaUVWMap<FT, IT> antenna_uvw;
    // Create numpy array holding the antenna coordinates
    py::array_t<FT, flags> result({int(ntime), int(nr_of_antenna), 3});

    // Find antenna UVW coordinates for each time chunk
    for(IT t=0, start=0; t<ntime; start += *time_chunks.data(t), ++t)
    {
        IT length = *time_chunks.data(t);

        // Loop twice
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, start, start+length);
        _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw, start, start+length);

        for(IT a=0; a<nr_of_antenna; ++a)
        {
            auto ant = antenna_uvw.find(a);

            // Not there, nan the antenna UVW coord
            if(ant == antenna_uvw.end())
            {
                *result.mutable_data(t, a, 0) = std::numeric_limits<FT>::quiet_NaN();
                *result.mutable_data(t, a, 1) = std::numeric_limits<FT>::quiet_NaN();
                *result.mutable_data(t, a, 2) = std::numeric_limits<FT>::quiet_NaN();
            }
            // Set the antenna UVW coordinate
            else
            {
                *result.mutable_data(t, a, 0) = ant->second.u;
                *result.mutable_data(t, a, 1) = ant->second.v;
                *result.mutable_data(t, a, 2) = ant->second.w;
            }

        }

        antenna_uvw.clear();
    }

    return result;
}


PYBIND11_MODULE(dataset_mod, m) {
    m.doc() = "auto-compiled c++ extension";

    m.def("antenna_uvw", &antenna_uvw<float, std::int32_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<float, std::int32_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<double, std::int64_t>, py::return_value_policy::move);
    m.def("antenna_uvw", &antenna_uvw<double, std::int64_t>, py::return_value_policy::move);
}