#ifndef _SEGREDUCE_H
#define _SEGREDUCE_H

#ifdef __cplusplus
extern "C" {
#endif 

void initsegreduce(void);

PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args);

#ifdef __cplusplus
} // extern "C" {}
#endif 

#endif