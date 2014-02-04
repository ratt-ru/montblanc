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
	{"segmented_reduce_complex128_sum", (PyCFunction) segmented_reduce_complex128_sum, METH_VARARGS | METH_KEYWORDS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initsegreduce()  {
	(void) Py_InitModule("segreduce", segreduce_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args, PyObject * kw)
{
	PyObject * value_array;		// pycuda.gpuarray
	PyObject * segment_starts;	// pycuda.gpuarray
	PyObject * segment_sums;	// pycuda.gpuarray
	int device_id;				// int
	PyObject * stream;			// pycuda.driver.Stream

    static char *kwlist[] = {
    	"value_array",
    	"segment_starts",
    	"segment_sums",
    	"device_id",
    	"stream",
    	NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO|iO", kwlist,
		&value_array,
		&segment_starts,
		&segment_sums,
		&device_id,
		&stream)) return NULL;

	PyObject * value_gpu = PyObject_GetAttrString(value_array, "gpudata");
	PyObject * value_size =  PyObject_GetAttrString(value_array, "size");
	PyObject * segments_gpu = PyObject_GetAttrString(segment_starts, "gpudata");
	PyObject * segments_size =  PyObject_GetAttrString(segment_starts, "size");
	PyObject * segment_sums_gpu = PyObject_GetAttrString(segment_sums, "gpudata");
	PyObject * stream_handle = PyObject_GetAttrString(stream, "handle"); 

	// Extract cuda device pointers, array sizes and stream_id
	// from the Python Objects
	CUdeviceptr value_ptr = (CUdeviceptr) value_gpu;
	CUdeviceptr segment_ptr = (CUdeviceptr) segments_gpu;
	CUdeviceptr segment_sums_ptr = (CUdeviceptr) segment_sums_gpu;
	long n_values =  PyInt_AsLong(value_size);
	long n_segments =  PyInt_AsLong(segments_size);
	CUstream stream_id = (CUstream) PyInt_AsLong(stream_handle);

	printf("values address=%p size=%ld\n", value_ptr, n_values);

	printf("segments address=%p size=%ld\n", segment_ptr, n_segments);

	printf("segment sums address=%p\n", segment_sums_ptr);

	printf("device_id=%ld stream=%ld\n", device_id, stream_id);

	// We've finished using all these Python Objects.
	Py_DECREF(value_gpu);
	Py_DECREF(value_size);
	Py_DECREF(segments_gpu);
	Py_DECREF(segments_size);
	Py_DECREF(segment_sums_gpu);
	Py_DECREF(stream_handle);

	mgpu::ContextPtr context_ptr = mgpu::CreateCudaDeviceAttachStream(
		device_id, stream_id);

	mgpu::SegReduceCsr(
		(double2 *) value_ptr,
		(int *) segment_ptr,
		n_values,
		n_segments,
		false,
		(double2 *) segment_sums_ptr,
		make_double2(0.,0.),
		mgpu::plus<double2>(),
		*context_ptr);

	return value_array;
}

#ifdef __cplusplus
} // extern "C" {
#endif 