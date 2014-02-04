INCLUDES = \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy \
	-Imoderngpu/include
CFLAGS = $(INCLUDES)

GENCODE_SM20 := -gencode arch=compute_20,code=sm_20
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

OBJECTS = segreduce.o mgpucontext.o mgpuutil.o
SHARED_LIBRARY = segreduce.so

# ---- Link --------------------------- 
$(SHARED_LIBRARY):  $(OBJECTS)
	nvcc  --compiler-options -fPIC -shared $(OBJECTS) -o segreduce.so -L/usr/lib/nvidia-331 -lcuda 

# ---- gcc C compile ------------------
segreduce.o:  segreduce.cu segreduce.h
	nvcc --compiler-options -fPIC  -c segreduce.cu $(CFLAGS) $(GENCODE_FLAGS)

mgpucontext.o: moderngpu/src/mgpucontext.cu
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpucontext.cu $(CFLAGS) $(GENCODE_FLAGS) 

mgpuutil.o: moderngpu/src/mgpuutil.cpp
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpuutil.cpp $(CFLAGS) $(GENCODE_FLAGS) 

clean:
	rm $(OBJECTS)