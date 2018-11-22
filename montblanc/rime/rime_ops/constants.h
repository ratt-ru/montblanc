#ifndef RIME_CONSTANTS_H
#define RIME_CONSTANTS_H

namespace montblanc {

template <typename FT>
struct constants
{
    static const FT lightspeed;
    static const FT pi;
    static const FT fwhm2int;
    static const FT sqrt_two;
    static const FT two_pi_over_c;
    static const FT gauss_scale;
};

}

#endif