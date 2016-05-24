TF_INC=$(shell python -c 'import tensorflow as tf; print tf.sysconfig.get_include()')
TF_CUDA=$(shell python -c 'import tensorflow as tf; print int(tf.test.is_built_with_cuda())')
MB_INC=../../include

INCLUDES = -I $(TF_INC) -I $(MB_INC)
OBJECTS=abstraction.o phase_op_cpu.o phase_op_gpu.o b_sqrt_op_cpu.o b_sqrt_op_gpu.o
LIBRARY=rime.so

CPPFLAGS=-std=c++11 -fPIC $(INCLUDES)
NVCCFLAGS=-std=c++11 -D GOOGLE_CUDA=$(TF_CUDA) $(INCLUDES) -x cu -Xcompiler -fPIC --gpu-architecture=sm_35

all : $(LIBRARY)

clean :
	rm -f $(OBJECTS) $(LIBRARY)

abstraction.o : abstraction.cu
	nvcc $< -c -o $@ $(NVCCFLAGS)

b_sqrt_op_gpu.o: b_sqrt_op_gpu.cu b_sqrt_op_gpu.cuh b_sqrt_op.h
	nvcc $< -c -o $@ $(NVCCFLAGS)

b_sqrt_op_cpu.o: b_sqrt_op_cpu.cpp b_sqrt_op_cpu.h b_sqrt_op.h
	g++ $< -c -o $@ $(CPPFLAGS)

phase_op_gpu.o: phase_op_gpu.cu phase_op_gpu.cuh phase_op.h
	nvcc $< -c -o $@ $(NVCCFLAGS)

phase_op_cpu.o : phase_op_cpu.cpp phase_op.h
	g++ $< -c -o $@ $(CPPFLAGS)

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) -fPIC