#ifndef RIME_SUM_COHERENCIES_OP_H
#define RIME_SUM_COHERENCIES_OP_H

#include "rime_constant_structures.h"

namespace montblanc {
namespace sumcoherencies {

template <typename Device, typename FT, typename CT> class RimeSumCoherencies;

typedef struct {
    dim_field nsrc;
    int ntime;
    int nbl;
    int na;
    int nchan;
    int npolchan;
    int npsrc;
    int nssrc;
    int ngsrc;

} const_data;

} // namespace sumcoherencies {
} // namespace montblanc {

#endif // #define RIME_SUM_COHERENCIES_OP_H