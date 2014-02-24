#ifndef _CRIMES_H
#define _CRIMES_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "arrayobject.h"
#include <cmath>
#include <cassert>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif 

void initcrimes(void);

static PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args,PyObject * kw);
static PyObject * segmented_reduce_complex64_sum(PyObject * self, PyObject * args,PyObject * kw);
static PyObject * segmented_reduce_float64_sum(PyObject * self, PyObject * args,PyObject * kw);
static PyObject * segmented_reduce_float32_sum(PyObject * self, PyObject * args,PyObject * kw);

#ifdef __cplusplus
} // extern "C" {}
#endif 

#endif