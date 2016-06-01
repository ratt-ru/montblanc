#ifndef RIME_SUM_COHERENCIES_OP_H
#define RIME_SUM_COHERENCIES_OP_H

#include "rime_constant_structures.h"

namespace montblanc {
namespace sumcoherencies {

template <typename Device, typename FT, typename CT> class RimeSumCoherencies;

typedef struct {
    dim_field nsrc;
    dim_field npsrc;
    dim_field ngsrc;
    dim_field nssrc;
    uint32_t ntime;
    uint32_t nbl;
    uint32_t na;
    uint32_t nchan;
    uint32_t npolchan;

} const_data;

} // namespace sumcoherencies {
} // namespace montblanc {

#endif // #define RIME_SUM_COHERENCIES_OP_H