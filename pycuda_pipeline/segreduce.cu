/* A file to test importing C modules for handling arrays to Python */

#include "Python.h"
#include "arrayobject.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "segreduce.h"

#include <cuda.h>
#include <moderngpu.cuh>

// Need a plus operator for this
inline __host__ __device__ double2 operator+(const double2 & lhs, const double2 & rhs)
	{ return make_double2(lhs.x + rhs.x, rhs.y + lhs.y); }

inline __host__ __device__ double2 & operator+=(double2 & lhs, const double2 & rhs)
	{ lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }


__global__ void dumb_kernel(int * data, int N)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= N)
		return;

	data[i] *= 2;
}

template <typename InputIt, typename CsrIt, typename OutputIt, typename T, typename Op>
void seg_reduce_csr_expand(InputIt data_global, CsrIt csr_global, int count,
	int numSegments, OutputIt dest_global, T identity, Op op, CUstream stream)
{
	// SegReduceHost (segreducecsr.cuh) STARTS
	typedef typename mgpu::SegReduceNormalTuning<sizeof(T)>::Tuning Tuning;

	// SegReduceInner (segreducecsr.cuh) STARTS
	// TODO: pass the PTX in from PyCUDA somehow
	int2 launch = Tuning::GetLaunchParams(300);
	int NV = launch.x * launch.y;
	int numBlocks = MGPU_DIV_UP(count, NV);
	const int * sources_global = (const int *) 0;

	// PartitionCsrSegReduce (segreduce.cuh) starts here
	int * limitsDevice;

	{
		int numPartitions = numBlocks + 1;
		int numRows = numSegments;
		const int * numRows2 = (const int *) 0;
		cudaMalloc(&limitsDevice, sizeof(int)*numPartitions);
		const int NT = 64;

		int numBlocks2 = MGPU_DIV_UP(numPartitions, NT);

		mgpu::KernelPartitionCsrSegReduce<NT><<<numBlocks2, NT, 0, stream>>>(
			count, NV, csr_global, numRows, numRows2, numPartitions,
			limitsDevice);
		// TODO: Add kernel error checking here
	}
	// PartitionCsrSegReduce (segreduce.cuh) ends here

	T * carryOutDevice;
	cudaMalloc(&carryOutDevice, sizeof(T)*numBlocks);

	mgpu::KernelSegReduceCsr<Tuning, false>
		<<<numBlocks, launch.x, 0, stream>>>(csr_global,
		sources_global, count, limitsDevice,
		data_global, identity, op, 
		dest_global, carryOutDevice);
	// TODO: Add kernel error checking here

	// SegReduceSpine (segreduce.cuh) starts here
	{
		const int NT = 128;
		int count = numBlocks;
		// redefine numBlocks, but RAII saves us.
		int numBlocks = MGPU_DIV_UP(count, NT);
		int * limits_global = limitsDevice;
		T * carryIn_global = carryOutDevice;
		// redefine carryOutDevice, but RAII saves us.
		T * carryOutDevice;
		cudaMalloc(&carryOutDevice, sizeof(T)*numBlocks);

		// Fix-up the segment outputs between the original tiles.
		mgpu::KernelSegReduceSpine1<NT><<<numBlocks, NT, 0, stream>>>(
			limits_global, count, dest_global, carryIn_global, identity, op,
			carryOutDevice);
		// TODO: Add kernel error checking here

		// Loop over the segments that span the tiles of 
		// KernelSegReduceSpine1 and fix those.
		if(numBlocks > 1) {
			mgpu::KernelSegReduceSpine2<NT><<<1, NT, 0, stream>>>(
				limits_global, numBlocks, count, NT, dest_global,
				carryOutDevice, identity, op);
		// TODO: Add kernel error checking here
		}

		cudaFree(carryOutDevice);
	}
	// SegReduceSpine (segreduce.cuh) ends here

	cudaFree(carryOutDevice);
	cudaFree(limitsDevice);

	// SegReduceInner (segreducecsr.cuh) ENDS
} 

template <typename T, typename Op>
PyObject * extract_and_segment(PyObject * self, PyObject * args, PyObject * kw,
	T identity, Op op)
{
	PyObject * value_array;		// pycuda.gpuarray
	PyObject * segment_starts;	// pycuda.gpuarray
	PyObject * segment_sums;	// pycuda.gpuarray
	int device_id;				// int
	PyObject * stream_obj;		// pycuda.driver.Stream

    static char * kwlist[] = {
    	(char *) "value_array",
    	(char *) "segment_starts",
    	(char *) "segment_sums",
    	(char *) "device_id",
    	(char *) "stream",
    	NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOiO", kwlist,
		&value_array,
		&segment_starts,
		&segment_sums,
		&device_id,
		&stream_obj)) return NULL;

	PyObject * value_gpu = PyObject_GetAttrString(value_array, "gpudata");
	PyObject * value_size =  PyObject_GetAttrString(value_array, "size");
	PyObject * segments_gpu = PyObject_GetAttrString(segment_starts, "gpudata");
	PyObject * segments_size =  PyObject_GetAttrString(segment_starts, "size");
	PyObject * segment_sums_gpu = PyObject_GetAttrString(segment_sums, "gpudata");
	PyObject * stream_handle = PyObject_GetAttrString(stream_obj, "handle"); 

	// Extract cuda device pointers, array sizes and stream_id
	// from the Python Objects
	T * value_ptr = (T *) PyInt_AsUnsignedLongLongMask(value_gpu);
	int * segment_ptr = (int *) PyInt_AsUnsignedLongLongMask(segments_gpu);
	T * segment_sums_ptr = (T *) PyInt_AsUnsignedLongLongMask(segment_sums_gpu);
	int n_values =  PyInt_AsLong(value_size);
	int n_segments =  PyInt_AsLong(segments_size);
	CUstream stream = (CUstream) PyInt_AsUnsignedLongLongMask(stream_handle);

	printf("values address=%p size=%ld\n", value_ptr, n_values);
	printf("segments address=%p size=%ld\n", segment_ptr, n_segments);
	printf("segment sums address=%p\n", segment_sums_ptr);
	printf("device_id=%ld stream=%ld\n", device_id, stream);

	seg_reduce_csr_expand(value_ptr, segment_ptr, n_values,
		n_segments, segment_sums_ptr, identity, op, stream);

	return value_array;
}

#ifdef __cplusplus
extern "C" {
#endif 

/* #### Globals #################################### */
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef segreduce_Methods[] = {
	{"segmented_reduce_complex128_sum", (PyCFunction) 		segmented_reduce_complex128_sum, METH_VARARGS | METH_KEYWORDS},
	{"segmented_reduce_float32_sum", (PyCFunction) 		segmented_reduce_float32_sum, METH_VARARGS | METH_KEYWORDS},
	{NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _C_arraytest in compile and linked 
void initsegreduce()  {
	(void) Py_InitModule("segreduce", segreduce_Methods);
	import_array();  // Must be present for NumPy.  Called first after above line.
}

static PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args,PyObject * kw)
{
	return extract_and_segment(self, args, kw,
		make_double2(0.,0.), mgpu::plus<double2>());	
}

static PyObject * segmented_reduce_float32_sum(PyObject * self, PyObject * args,PyObject * kw)
{
	return extract_and_segment(self, args, kw,
		0.0f, mgpu::plus<float>());
}

#ifdef __cplusplus
} // extern "C" {
#endif 