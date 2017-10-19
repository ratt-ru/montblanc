/*
<%
setup_pybind11(cfg)
cfg['compiler_args'] = ['-std=c++11', '-fvisibility=hidden']
cfg['libraries'] = ['casa_casa', 'casa_measures']
%>
*/

#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MPosition.h>
#include <casacore/measures/Measures/MeasConvert.h>
#include <casacore/measures/Measures/MeasTable.h>
#include <casacore/measures/Measures/MEpoch.h>

namespace py = pybind11;

constexpr unsigned int flags = py::array::c_style | py::array::forcecast;

template <typename FT>
py::array_t<FT, flags> parallactic_angles(
    py::array_t<FT, flags> times,
    py::array_t<FT, flags> antenna_positions,
    py::array_t<FT, flags> phase_centre)
{
    py::gil_scoped_release release;

    int na = antenna_positions.shape(0);
    int ntimes = times.shape(0);

    // Result array
    py::array_t<FT, flags> angles({ntimes, na});

    std::vector<casa::MPosition> itrf_antenna_positions;
    itrf_antenna_positions.reserve(na);

    // Compute antenna positions in ITRF
    for(int ant=0; ant<na; ++ant)
    {
        const FT * x = antenna_positions.data(ant, 0);
        const FT * y = antenna_positions.data(ant, 1);
        const FT * z = antenna_positions.data(ant, 2);

        itrf_antenna_positions.push_back(casa::MPosition(
                casa::MVPosition(*x, *y, *z),
                casa::MPosition::ITRF));
    }

    // Direction towards zenith
    casa::MVDirection base_zenith(0, M_PI/2);

    // Direction towards phase centre
    casa::MVDirection phase_dir(*phase_centre.data(0), *phase_centre.data(1));

    // For each time
    for(int time=0; time<ntimes; ++time)
    {
        // Create a frame for this timestemp
        casa::MeasFrame frame(casa::MEpoch(
            casa::Quantum<double>(*times.data(time), "s"),
            casa::MEpoch::UTC));

        // For each antenna
        for(int ant=0; ant<na; ++ant)
        {
            // Set the frame's position to the antenna position
            frame.set(itrf_antenna_positions[ant]);

            // Direction to the zenith in this frame
            casa::MDirection mzenith(base_zenith, casa::MDirection::Ref(
                casa::MDirection::AZELGEO, frame));

            // Compute parallactic angle of phase direction w.r.t zenith
            casa::MeasConvert<casa::MDirection> convert(mzenith, casa::MDirection::J2000);
            *angles.mutable_data(time, ant) = phase_dir.positionAngle(convert().getValue());
        }
    }

    return angles;
}

PYBIND11_MODULE(casa_parangles, m) {
    m.doc() = "auto-compiled c++ extension";

    m.def("parallactic_angles", &parallactic_angles<float>, py::return_value_policy::move);
    m.def("parallactic_angles", &parallactic_angles<double>, py::return_value_policy::move);
}