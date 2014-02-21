NVCC=nvcc

SEGREDUCE_INCLUDES = \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy \
	-Imoderngpu/include
SEGREDUCE_CFLAGS = $(SEGREDUCE_INCLUDES)

SEGREDUCE_LIBDIRS=-L/usr/lib/nvidia-331
SEGREDUCE_LIBS = -lcuda
SEGREDUCE_LDFLAGS = $(SEGREDUCE_LIBDIRS) $(SEGREDUCE_LIBS) 

GENCODE_SM20 := -gencode arch=compute_20,code=sm_20
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

SEGREDUCE_OBJECTS = segreduce.o mgpucontext.o mgpuutil.o
SEGREDUCE_LIBRARY = segreduce.so

CC=gcc

PREDICT_INCLUDES = \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy
PREDICT_CFLAGS = $(PREDICT_INCLUDES) -fopenmp -fPIC

PREDICT_LIBDIRS =
PREDICT_LIBS = -lgomp
PREDICT_LDFLAGS = $(PREDICT_LIBDIRS) $(PREDICT_LIBS) -fopenmp

PREDICT_OBJECTS = predict.o
PREDICT_LIBRARY = predict.so

all: $(PREDICT_LIBRARY) $(SEGREDUCE_LIBRARY) 

# ---- Link --------------------------- 
$(PREDICT_LIBRARY):  $(PREDICT_OBJECTS)
	$(CC) -fPIC -shared $(PREDICT_OBJECTS) -o $(PREDICT_LIBRARY) $(PREDICT_LDFLAGS)

$(SEGREDUCE_LIBRARY):  $(SEGREDUCE_OBJECTS)
	$(NVCC)  --compiler-options -fPIC -shared $(SEGREDUCE_OBJECTS) -o $(SEGREDUCE_LIBRARY) $(SEGREDUCE_LDFLAGS)

# ---- gcc C compile ------------------
predict.o:  predict.c predict.h
	$(CC) predict.c -c $(PREDICT_CFLAGS)

segreduce.o:  segreduce.cu segreduce.h
	nvcc --compiler-options -fPIC  -c segreduce.cu $(SEGREDUCE_CFLAGS) $(GENCODE_FLAGS)

mgpucontext.o: moderngpu/src/mgpucontext.cu
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpucontext.cu $(SEGREDUCE_CFLAGS) $(GENCODE_FLAGS) 

mgpuutil.o: moderngpu/src/mgpuutil.cpp
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpuutil.cpp $(SEGREDUCE_CFLAGS) $(GENCODE_FLAGS) 

clean:
	rm $(SEGREDUCE_OBJECTS) $(PREDICT_OBJECTS)