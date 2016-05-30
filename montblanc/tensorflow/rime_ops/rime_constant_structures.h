#ifndef RIME_CONSTANT_STRUCTURES_H
#define RIME_CONSTANT_STRUCTURES_H

#include <cstdint>

namespace montblanc {

typedef struct {
    uint32_t global_size;
    uint32_t local_size;
    uint32_t lower_extent;
    uint32_t upper_extent;
    
} dim_field;

typedef struct {
    dim_field ntime;
    /*
    dim_field na;
    dim_field nbl;
    dim_field nchan;
    dim_field npolchan;
    dim_field beam_lw;
    dim_field beam_mh;
    dim_field beam_nud;
    */

} rime_const_data;

} //namespace montblanc

#endif