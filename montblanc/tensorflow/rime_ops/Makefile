TF_INC=$(shell python -c 'import tensorflow as tf; print tf.sysconfig.get_include()')
TF_CUDA=$(shell python -c 'import tensorflow as tf; print int(tf.test.is_built_with_cuda())')

LIBRARY=rime.so

all : $(LIBRARY)

clean :
	rm $(LIBRARY)

phase_op_gpu.cu.o: phase_op_gpu.cu.cc
	echo $(GOOLE_CUDA)
	nvcc -std=c++11 -c -o phase_op_gpu.cu.o phase_op_gpu.cu.cc -I $(TF_INC) -D GOOGLE_CUDA=$(TF_CUDA) -x cu -Xcompiler -fPIC

phase_op.o : phase_op.cc
	g++ phase_op.cc --std=c++11 -I $(TF_INC) -c -fPIC -D GOOGLE_CUDA=$(TF_CUDA)

$(LIBRARY) : phase_op.o phase_op_gpu.cu.o
	g++  -shared phase_op.o -o $(LIBRARY) -fPIC