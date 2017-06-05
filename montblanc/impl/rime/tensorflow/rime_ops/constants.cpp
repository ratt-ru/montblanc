// For M_PI
#define _USE_MATH_DEFINES
#include <cmath>

#include "constants.h"

namespace montblanc {

template <> const float
constants<float>::lightspeed = 299792458.0;
template <> const double
constants<double>::lightspeed = 299792458.0;

template <> const float
constants<float>::pi = M_PI;
template <> const double
constants<double>::pi = M_PI;

template <> const float
constants<float>::sqrt_two = std::sqrt(2.0);

template <> const double
constants<double>::sqrt_two = std::sqrt(2.0);

template <> const float
constants<float>::fwhm2int = 1.0/std::sqrt(std::log(256.0));
template <> const double
constants<double>::fwhm2int = 1.0/std::sqrt(std::log(256.0));

template <> const float
constants<float>::two_pi_over_c =
    2.0f*constants<float>::lightspeed/constants<float>::pi;
template <> const double
constants<double>::two_pi_over_c =
    2.0f*constants<double>::lightspeed/constants<double>::pi;

template <> const float
constants<float>::gauss_scale = (constants<float>::fwhm2int*constants<float>::sqrt_two
    *constants<float>::pi)/constants<float>::lightspeed;

template <> const double
constants<double>::gauss_scale = (constants<double>::fwhm2int*constants<double>::sqrt_two
    *constants<double>::pi)/constants<double>::lightspeed;

}
