TF_INC=$(shell python -c 'import tensorflow as tf; print tf.sysconfig.get_include()')

LIBRARY=rime.so

all : $(LIBRARY)

clean :
	rm $(LIBRARY)

$(LIBRARY) :  phase_op.cc
	g++ phase_op.cc --std=c++11 -shared -I $(TF_INC) -o $(LIBRARY) -fPIC