# Tensorflow includes and defines
TF_INC=$(shell python -c 'import tensorflow as tf; print tf.sysconfig.get_include()')
TF_CUDA=$(shell python -c 'import tensorflow as tf; print int(tf.test.is_built_with_cuda())')
MB_INC=../../include

# Dependencies
DEPDIR:=.d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS=-MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

# Define our sources, objects and libraries
SOURCES=$(wildcard *.cpp *.cu)
OBJECTS=$(addsuffix .o, $(basename $(SOURCES)))
LIBRARY=rime.so

# Compiler flags
INCLUDES= -I $(TF_INC) -I $(MB_INC)
CPPFLAGS=-std=c++11 $(INCLUDES) -fPIC 
NVCCFLAGS=-std=c++11 -D GOOGLE_CUDA=$(TF_CUDA) $(INCLUDES) \
	-x cu --compiler-options "-fPIC" --gpu-architecture=sm_35

# Compiler directives
COMPILE.cpp = g++ $(DEPFLAGS) $(CPPFLAGS) -c
COMPILE.nvcc = nvcc --compiler-options " $(DEPFLAGS)" $(NVCCFLAGS) -c

all : $(LIBRARY)

%.o : %.cpp
	$(COMPILE.cpp) $<

%.o : %.cu
	$(COMPILE.nvcc) $<

clean :
	rm -f $(OBJECTS) $(LIBRARY)

$(LIBRARY) : $(OBJECTS)
	g++  -shared $(OBJECTS) -o $(LIBRARY) -fPIC

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))	
