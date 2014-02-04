/* A file to test importing C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "segreduce.h"

#include <moderngpu.cuh>

// Need a plus operator for this
inline __host__ __device__ double2 operator+(const double2 & lhs, const double2 & rhs)
	{ return make_double2(lhs.x + rhs.x, rhs.y + lhs.y); }

inline __host__ __device__ double2 & operator+=(double2 & lhs, const double2 & rhs)
	{ lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }


#ifdef __cplusplus
extern "C" {
#endif 

/* #### Globals #################################### */
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef segreduce_Methods[] = {
	{"segmented_reduce_complex128_sum", segmented_reduce_complex128_sum, METH_VARARGS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initsegreduce()  {
	(void) Py_InitModule("segreduce", segreduce_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args)
{
	PyObject * value_array;
	PyObject * segment_starts;
	PyObject * segment_sums;
	PyObject * cuda_device_id;
	PyObject * cuda_stream;

	if(!PyArg_ParseTuple(args, "OOOOO",
		&value_array,
		&segment_starts,
		&segment_sums,
		&cuda_device_id,
		&cuda_stream)) return NULL;

	CUdeviceptr value_ptr = (CUdeviceptr) PyObject_GetAttrString(value_array, "gpudata");
	int * size =  (int *) PyObject_GetAttrString(value_array, "size");

	printf("address %d size %d\n", value_ptr, *size);

	CUdeviceptr segment_start_ptr = NULL;
	CUdeviceptr segment_sum_ptr = NULL;

	int n_values = 1024*1024;
	int n_segments = 10;

	CUstream stream = 0;


	mgpu::ContextPtr context_ptr = mgpu::CreateCudaDeviceAttachStream(1, stream);

	mgpu::SegReduceCsr(
		(double2 *) value_ptr,
		(int *) segment_start_ptr,
		n_values,
		n_segments,
		false,
		(double2 *) segment_sum_ptr,
		make_double2(0.,0.),
		mgpu::plus<double2>(),
		*context_ptr);

	return value_array;
}

#ifdef __cplusplus
} // extern "C" {
#endif 