#ifndef RIME_CONSTANT_STRUCTURES_H
#define RIME_CONSTANT_STRUCTURES_H

#include <cstdint>

namespace montblanc {

typedef struct {
    int global_size;
    int local_size;
    int lower_extent;
    int upper_extent;

#ifdef GOOGLE_CUDA
    __host__ __device__ __forceinline__
#else
    inline
#endif
    int extent_size(void) const
        { return upper_extent - lower_extent; }
    
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