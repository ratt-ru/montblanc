#ifndef _SEGREDUCE_H
#define _SEGREDUCE_H

#ifdef __cplusplus
extern "C" {
#endif 

void initsegreduce(void);

static PyObject * segmented_reduce_complex128_sum_expanded(PyObject * self, PyObject * args,PyObject * kw);
static PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args, PyObject * kw);
static PyObject * segmented_reduce_float32_sum(PyObject * self, PyObject * args,PyObject * kw);

#ifdef __cplusplus
} // extern "C" {}
#endif 

#endif