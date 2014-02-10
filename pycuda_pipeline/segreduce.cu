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

#ifdef __cplusplus
extern "C" {
#endif 

/* #### Globals #################################### */
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */

/* ==== Set up the methods table ====================== */
static PyMethodDef segreduce_Methods[] = {
	{"segmented_reduce_complex128_sum_expanded", (PyCFunction) 						segmented_reduce_complex128_sum_expanded, METH_VARARGS | METH_KEYWORDS},
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

static PyObject * segmented_reduce_complex128_sum_expanded(PyObject * self, PyObject * args,PyObject * kw)
{
	PyObject * value_array;		// pycuda.gpuarray
	PyObject * segment_starts;	// pycuda.gpuarray
	PyObject * segment_sums;	// pycuda.gpuarray
	int device_id;				// int
	PyObject * stream_obj;		// pycuda.driver.Stream
	PyObject * context_obj;		// pycuda.driver.Context
	unsigned long long test_ptr;

    static char *kwlist[] = {
    	"value_array",
    	"segment_starts",
    	"segment_sums",
    	"device_id",
    	"stream",
    	"context",
    	"test_ptr",
    	NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOiOOK", kwlist,
		&value_array,
		&segment_starts,
		&segment_sums,
		&device_id,
		&stream_obj,
		&context_obj,
		&test_ptr)) return NULL;

	PyObject * value_gpu = PyObject_GetAttrString(value_array, "gpudata");
	PyObject * value_size =  PyObject_GetAttrString(value_array, "size");
	PyObject * segments_gpu = PyObject_GetAttrString(segment_starts, "gpudata");
	PyObject * segments_size =  PyObject_GetAttrString(segment_starts, "size");
	PyObject * segment_sums_gpu = PyObject_GetAttrString(segment_sums, "gpudata");
	PyObject * stream_handle = PyObject_GetAttrString(stream_obj, "handle"); 
//	PyObject * context_handle = PyObject_GetAttrString(context_obj, "handle"); 

	// Extract cuda device pointers, array sizes and stream_id
	// from the Python Objects
	double2 * value_ptr = (double2 *) PyInt_AsUnsignedLongLongMask(value_gpu);
	int * segment_ptr = (int *) PyInt_AsUnsignedLongLongMask(segments_gpu);
	double2 * segment_sums_ptr = (double2 *) PyInt_AsUnsignedLongLongMask(segment_sums_gpu);
	int n_values =  PyInt_AsLong(value_size);
	int n_segments =  PyInt_AsLong(segments_size);
	CUstream stream = (CUstream) PyInt_AsUnsignedLongLongMask(stream_handle);
//	CUcontext context = (CUcontext) PyInt_AsUnsignedLongLongMask(context_handle);

	printf("values address=%p size=%ld\n", value_ptr, n_values);
	printf("segments address=%p size=%ld\n", segment_ptr, n_segments);
	printf("segment sums address=%p\n", segment_sums_ptr);
	printf("device_id=%ld stream=%ld\n", device_id, stream);
	printf("test_ptr=%lu values address=%lu\n", test_ptr, value_ptr);

	seg_reduce_csr_expand(value_ptr, segment_ptr, n_values,
		n_segments, segment_sums_ptr, make_double2(0.,0.), mgpu::plus<double2>(), stream);

	/*
	typedef typename mgpu::SegReduceNormalTuning<sizeof(double2)>::Tuning Tuning;

	mgpu::plus<double2> op;
	double2 id = make_double2(0.,0.);

	// TODO: pass the PTX in from PyCUDA somehow
	int2 launch = Tuning::GetLaunchParams(300);
	int nv = launch.x * launch.y;
	int n_blocks = MGPU_DIV_UP(n_values, nv);
	int n_partitions =  n_blocks + 1;
	const int nt = 64;

	int * limits_ptr;
	double2 * carry_out_ptr;
	cudaMalloc(&limits_ptr, sizeof(int)*n_partitions);
	cudaMalloc(&carry_out_ptr, sizeof(double2)*n_blocks);

	// Use upper-bound binary search to partition the CSR structure into tiles.
	//MGPU_MEM(int) limitsDevice = PartitionCsrSegReduce(count, nv, csr_global,
	//	numSegments, numSegments2_global, numBlocks + 1, context);

	int n_blocks_2 = MGPU_DIV_UP(n_partitions, nt);

	mgpu::KernelPartitionCsrSegReduce<nt><<<n_blocks_2, nt, 0, stream>>>(
		n_values, nv, segment_ptr, n_segments, (const int *) 0, n_partitions, limits_ptr);

	mgpu::KernelSegReduceCsr<Tuning, false>
		<<<n_blocks, launch.x, 0, stream>>>(segment_ptr,
		(const int *) 0, n_values, (const int *) limits_ptr, value_ptr,
		id, op, segment_sums_ptr, carry_out_ptr);

	{
		// SegReduceSpine from segreduce.cuh
		const int NT = 128;
		int n_blocks_spine = MGPU_DIV_UP(n_values, NT);
		int * carry_out_spine_ptr;
		cudaMalloc(&carry_out_spine_ptr, sizeof(int)*n_blocks_spine);

		// Fix-up the segment outputs between the original tiles.
//		MGPU_MEM(T) carryOutDevice = context.Malloc<T>(n_blocks_spine);
		mgpu::KernelSegReduceSpine1<NT><<<n_blocks_spine, NT, 0, stream>>>(
			limits_ptr, n_values, segment_sums_ptr, carry_out_ptr,
			id, op, carry_out_spine_ptr);

		// Loop over the segments that span the tiles of 
		// KernelSegReduceSpine1 and fix those.
		if(n_blocks_spine > 1) {
			mgpu::KernelSegReduceSpine2<NT><<<1, NT, 0, stream>>>(
				limits_ptr, n_blocks_spine, n_values, NT, segment_sums_ptr,
				carry_out_spine_ptr, id, op);
		}

		cudaFree(carry_out_spine_ptr);
	}

	cudaFree(limits_ptr);
	cudaFree(carry_out_ptr);
	*/

	/*
	{
		int * limits_ptr;
		int N = 1024*1024*128;

		cudaMalloc(&limits_ptr, sizeof(int)*N);

		int * limits_host = new int[N];

		cudaMemcpyAsync(limits_ptr,limits_host, sizeof(int)*N, cudaMemcpyHostToDevice, stream);

		int threads_per_block = 1024;

		dim3 grid(N/threads_per_block, 1, 1);
		dim3 block(threads_per_block, 1, 1);

		dumb_kernel<<<grid, block, 0, stream>>>(limits_ptr, N);

		cudaMemcpyAsync(limits_host,limits_ptr, sizeof(int)*N, cudaMemcpyDeviceToHost, stream);

		delete [] limits_host;
		cudaFree(limits_ptr);
	}
	*/


	return value_array;
}

static PyObject * segmented_reduce_complex128_sum(PyObject * self, PyObject * args,PyObject * kw)
{
	PyObject * value_array;		// pycuda.gpuarray
	PyObject * segment_starts;	// pycuda.gpuarray
	PyObject * segment_sums;	// pycuda.gpuarray
	int device_id;				// int
	PyObject * stream_obj;		// pycuda.driver.Stream
	PyObject * context_obj;		// pycuda.driver.Context
	unsigned long long test_ptr;

    static char *kwlist[] = {
    	"value_array",
    	"segment_starts",
    	"segment_sums",
    	"device_id",
    	"stream",
    	"context",
    	"test_ptr",
    	NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOiOOK", kwlist,
		&value_array,
		&segment_starts,
		&segment_sums,
		&device_id,
		&stream_obj,
		&context_obj,
		&test_ptr)) return NULL;

	PyObject * value_gpu = PyObject_GetAttrString(value_array, "gpudata");
	PyObject * value_size =  PyObject_GetAttrString(value_array, "size");
	PyObject * segments_gpu = PyObject_GetAttrString(segment_starts, "gpudata");
	PyObject * segments_size =  PyObject_GetAttrString(segment_starts, "size");
	PyObject * segment_sums_gpu = PyObject_GetAttrString(segment_sums, "gpudata");
	PyObject * stream_handle = PyObject_GetAttrString(stream_obj, "handle"); 
//	PyObject * context_handle = PyObject_GetAttrString(context_obj, "handle"); 

	// Extract cuda device pointers, array sizes and stream_id
	// from the Python Objects
	CUdeviceptr value_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(value_gpu);
	CUdeviceptr segment_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(segments_gpu);
	CUdeviceptr segment_sums_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(segment_sums_gpu);
	long n_values =  PyInt_AsLong(value_size);
	long n_segments =  PyInt_AsLong(segments_size);
	CUstream stream = (CUstream) PyInt_AsUnsignedLongLongMask(stream_handle);
//	CUcontext context = (CUcontext) PyInt_AsUnsignedLongLongMask(context_handle);

	printf("values address=%p size=%ld\n", value_ptr, n_values);
	printf("segments address=%p size=%ld\n", segment_ptr, n_segments);
	printf("segment sums address=%p\n", segment_sums_ptr);
	printf("device_id=%ld stream=%ld\n", device_id, stream);
	printf("test_ptr=%lu values address=%lu\n", test_ptr, value_ptr);

//	cuCtxPushCurrent(context);

//	mgpu::ContextPtr context_ptr = mgpu::CreateCudaDeviceAttachStream(
//		device_id, stream_id);

	mgpu::ContextPtr context_ptr = mgpu::CreateCudaDevice(
		device_id);

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

//	cuCtxPopCurrent(&context);

	// We've finished using all these Python Objects.
	Py_DECREF(value_gpu);
	Py_DECREF(value_size);
	Py_DECREF(segments_gpu);
	Py_DECREF(segments_size);
	Py_DECREF(segment_sums_gpu);
	Py_DECREF(stream_handle);

	return value_array;
}

static PyObject * segmented_reduce_float32_sum(PyObject * self, PyObject * args,PyObject * kw)
{
	PyObject * value_array;		// pycuda.gpuarray
	PyObject * segment_starts;	// pycuda.gpuarray
	PyObject * segment_sums;	// pycuda.gpuarray
	int device_id;				// int
	PyObject * stream;			// pycuda.driver.Stream
	unsigned long long test_ptr;

    static char *kwlist[] = {
    	"value_array",
    	"segment_starts",
    	"segment_sums",
    	"device_id",
    	"stream",
    	"test_ptr",
    	NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOiOK", kwlist,
		&value_array,
		&segment_starts,
		&segment_sums,
		&device_id,
		&stream,
		&test_ptr)) return NULL;

	PyObject * value_gpu = PyObject_GetAttrString(value_array, "gpudata");
	PyObject * value_size =  PyObject_GetAttrString(value_array, "size");
	PyObject * segments_gpu = PyObject_GetAttrString(segment_starts, "gpudata");
	PyObject * segments_size =  PyObject_GetAttrString(segment_starts, "size");
	PyObject * segment_sums_gpu = PyObject_GetAttrString(segment_sums, "gpudata");
	PyObject * stream_handle = PyObject_GetAttrString(stream, "handle"); 

	// Extract cuda device pointers, array sizes and stream_id
	// from the Python Objects
	CUdeviceptr value_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(value_gpu);
	CUdeviceptr segment_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(segments_gpu);
	CUdeviceptr segment_sums_ptr = (CUdeviceptr) PyInt_AsUnsignedLongLongMask(segment_sums_gpu);
	long n_values =  PyInt_AsLong(value_size);
	long n_segments =  PyInt_AsLong(segments_size);
	CUstream stream_id = (CUstream) PyInt_AsUnsignedLongLongMask(stream_handle);

	printf("values address=%p size=%ld\n", value_ptr, n_values);
	printf("segments address=%p size=%ld\n", segment_ptr, n_segments);
	printf("segment sums address=%p\n", segment_sums_ptr);
	printf("device_id=%ld stream=%ld\n", device_id, stream_id);
	printf("test_ptr=%lu values address=%lu\n", test_ptr, value_ptr);

	//mgpu::ContextPtr context_ptr = mgpu::CreateCudaDeviceAttachStream(
	//	device_id, stream_id);

	mgpu::ContextPtr context_ptr = mgpu::CreateCudaDevice(
		device_id);

	mgpu::SegReduceCsr(
		(float *) value_ptr,
		(int *) segment_ptr,
		n_values,
		n_segments,
		false,
		(float *) segment_sums_ptr,
		float(0.0f),
		mgpu::plus<float>(),
		*context_ptr);

	// We've finished using all these Python Objects.
	Py_DECREF(value_gpu);
	Py_DECREF(value_size);
	Py_DECREF(segments_gpu);
	Py_DECREF(segments_size);
	Py_DECREF(segment_sums_gpu);
	Py_DECREF(stream_handle);

	return value_array;
}


#ifdef __cplusplus
} // extern "C" {
#endif 