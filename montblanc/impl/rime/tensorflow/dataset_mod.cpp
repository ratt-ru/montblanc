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
    AntennaUVWMap<FT, IT> & antenna_uvw)
{
    // Special case, infer the first (two) antenna coordinate(s)
    // from the first row
    if(antenna_uvw.size() == 0)
    {
        IT ant1 = *antenna1.data(0);
        IT ant2 = *antenna2.data(0);
        const FT * u = uvw.data(0,0);
        const FT * v = uvw.data(0,1);
        const FT * w = uvw.data(0,2);

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
    for(int row=1; row < antenna1.shape(0); ++row)
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
    py::array_t<IT, flags> antenna2)
{
    py::gil_scoped_release release;

    if(antenna1.ndim() != 1)
        { throw std::invalid_argument("antenna1 shape should be (nrow,)");}

    if(antenna2.ndim() != 1)
        { throw std::invalid_argument("antenna2 shape should be (nrow,)");}

    if(uvw.ndim() != 2 || uvw.shape(1) != 3)
        { throw std::invalid_argument("uvw shape should be (nrow, 3)");}

    AntennaUVWMap<FT, IT> antenna_uvw;

    // Loop twice
    _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw);
//    _antenna_uvw_loop(uvw, antenna1, antenna2, antenna_uvw);

    // Find the largest antenna number
    IT largest_ant = -1;

    for(const auto & ant: antenna_uvw)
        { largest_ant = std::max(largest_ant, ant.first); }

    if(largest_ant < 0)
        { throw std::invalid_argument("largest_ant < 0"); }

    // Create a numpy array holding the antenna coordinates
    py::array_t<FT, flags> result({int(largest_ant)+1, 3});

    for(IT i=0; i<largest_ant+1; ++i)
    {
        auto ant = antenna_uvw.find(i);

        // Not there, nan the antenna UVW coord
        if(ant == antenna_uvw.end())
        {
            *result.mutable_data(i, 0) = std::numeric_limits<FT>::quiet_NaN();
            *result.mutable_data(i, 1) = std::numeric_limits<FT>::quiet_NaN();
            *result.mutable_data(i, 2) = std::numeric_limits<FT>::quiet_NaN();
        }
        // Set the antenna UVW coordinate
        else
        {
            *result.mutable_data(i, 0) = ant->second.u;
            *result.mutable_data(i, 1) = ant->second.v;
            *result.mutable_data(i, 2) = ant->second.w;
        }
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