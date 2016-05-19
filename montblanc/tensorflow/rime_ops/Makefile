TF_INC=$(shell python -c 'import tensorflow as tf; print tf.sysconfig.get_include()')
TF_CUDA=$(shell python -c 'import tensorflow as tf; print int(tf.test.is_built_with_cuda())')
MB_INC=../../include

INCLUDES = -I $(TF_INC) -I $(TF_CUDA) -I $(MB_INC)
OBJECTS=phase_op.o phase_op_gpu.o b_sqrt_op_cpu.o b_sqrt_op_gpu.o
LIBRARY=rime.so

all : $(LIBRARY)

clean :
	rm $(OBJECTS) $(LIBRARY)


b_sqrt_op_gpu.o: b_sqrt_op_gpu.cu.cc b_sqrt_op_gpu.h b_sqrt_op.h
	nvcc -std=c++11 -c -o b_sqrt_op_gpu.o b_sqrt_op_gpu.cu.cc $(INCLUDES) -D GOOGLE_CUDA=$(TF_CUDA) -x cu -Xcompiler -fPIC

b_sqrt_op_cpu.o: b_sqrt_op_cpu.cc b_sqrt_op_cpu.h b_sqrt_op.h
	g++ b_sqrt_op_cpu.cc --std=c++11 $(INCLUDES) -c -fPIC

phase_op_gpu.o: phase_op_gpu.cu.cc phase_op_gpu.h phase_op.h
	nvcc -std=c++11 -c -o phase_op_gpu.o phase_op_gpu.cu.cc $(INCLUDES) -D GOOGLE_CUDA=$(TF_CUDA) -x cu -Xcompiler -fPIC

phase_op.o : phase_op.cc phase_op.h
	g++ phase_op.cc --std=c++11 $(INCLUDE) -c -fPIC

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) -fPIC