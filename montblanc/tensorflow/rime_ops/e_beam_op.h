#ifndef RIME_E_BEAM_OP_H_
#define RIME_E_BEAM_OP_H_

#include "rime_constant_structures.h"

namespace montblanc {
namespace ebeam {

template <typename Device, typename FT, typename CT> class EBeam;

// Number of polarisations handled by this kernel
constexpr int EBEAM_NPOL = 4;

template <typename FT>
struct const_data {
    int nsrc;
    int ntime;
    int na;
    int nchan;
    int npolchan;
    int npol;
    int beam_lw;
    int beam_mh;
    int beam_nud;
    FT ll, lm, lf;
    FT ul, um, uf;
};

} // namespace ebeam {
} // namespace montblanc {

#endif // #define RIME_E_BEAM_OP_H_