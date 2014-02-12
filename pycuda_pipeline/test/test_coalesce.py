import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np


KERNEL="""
struct __align__(16) double4c
{
    double2 a;
    double2 b;
    double2 c;
    double2 d;
};
typedef struct double4c double4c;

__global__ void dumb_kernel(double4c * m, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i >= N) return;

	double4c matrix = m[i];

	matrix.a.x *= 1.; matrix.a.y *= 1.;
	matrix.b.x *= 2.; matrix.b.y *= 2.;
	matrix.c.x *= 3.; matrix.c.y *= 3.;
	matrix.d.x *= 4.; matrix.d.y *= 4.;

	m[i] = matrix;
}
"""

mod = SourceModule(KERNEL, options=['-lineinfo'])
kernel = mod.get_function('dumb_kernel')

N = 1024
shape = (N,4)

threads_per_block = 512
blocks = (N + threads_per_block - 1) / threads_per_block

a = np.ones(np.product(shape)).astype(np.complex128).reshape(shape)
a_gpu = gpuarray.to_gpu(a)
kernel(a_gpu, np.int32(N),
	block=(threads_per_block,1,1), 
	grid=(blocks,1,1))

print a[:10,:]
print a_gpu.get()[:10,:]
