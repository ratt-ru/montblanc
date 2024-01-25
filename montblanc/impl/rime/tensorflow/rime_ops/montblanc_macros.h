#ifndef MONTBLANC_MACROS_H
#define MONTBLANC_MACROS_H

#ifdef DCOMPAT_TF2_4
    #define MB_STAT_OK Status::OK
#else
    #define MB_STAT_OK Status
#endif //DCOMPAT_TF2_4

#endif //MONTBLANC_MACROS_H