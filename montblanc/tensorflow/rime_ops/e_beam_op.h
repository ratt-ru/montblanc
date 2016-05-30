#ifndef RIME_E_BEAM_OP_H_
#define RIME_E_BEAM_OP_H_

namespace tensorflow {

template <typename Device, typename FT, typename CT> class RimeEBeam;

// Number of polarisations handled by this kernel
constexpr int EBEAM_NPOL = 4;

}  // namespace tensorflow

#endif // #define RIME_E_BEAM_OP_H_