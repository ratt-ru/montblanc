#ifndef MONTBLANC_OP_KERNEL_UTILS_H
#define MONTBLANC_OP_KERNEL_UTILS_H

#include <cuda.h>

#define OP_REQUIRES_CUDA_SUCCESS(CTX)                                   \
	do {                                                                \
		cudaError_t e = cudaGetLastError();                             \
		if(e != cudaSuccess)                                            \
		{                                                               \
		    (CTX)->CtxFailureWithWarning(__FILE__, __LINE__,            \
		    	::tensorflow::errors::Internal("Cuda Failure ",         \
		    		                           cudaGetErrorString(e))); \
		    return;                                                     \
		}                                                               \
	} while(0)

#endif
