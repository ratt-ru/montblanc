NVCC=nvcc

CRIMES_INCLUDES = \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy \
	-Imoderngpu/include
CRIMES_CFLAGS = $(CRIMES_INCLUDES) -O3

CRIMES_LIBDIRS=-L/usr/lib/nvidia-331
CRIMES_LIBS = -lcuda
CRIMES_LDFLAGS = $(CRIMES_LIBDIRS) $(CRIMES_LIBS) -O3

GENCODE_SM20 := -gencode arch=compute_20,code=sm_20
GENCODE_SM30 := -gencode arch=compute_30,code=sm_30
GENCODE_SM35 := -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS := $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

CRIMES_OBJECTS = crimes.o mgpucontext.o mgpuutil.o
CRIMES_LIBRARY = crimes.so

CC=gcc

PREDICT_INCLUDES = \
	-I/usr/include/python2.7 \
	-I/usr/lib/python2.7/dist-packages/numpy/core/include/numpy
PREDICT_CFLAGS = $(PREDICT_INCLUDES) -fopenmp -fPIC -O3

PREDICT_LIBDIRS =
PREDICT_LIBS = -lgomp
PREDICT_LDFLAGS = $(PREDICT_LIBDIRS) $(PREDICT_LIBS) -fopenmp -O3

PREDICT_OBJECTS = predict.o
PREDICT_LIBRARY = predict.so

all: $(PREDICT_LIBRARY) $(CRIMES_LIBRARY) 

# ---- Link --------------------------- 
$(PREDICT_LIBRARY):  $(PREDICT_OBJECTS)
	$(CC) -fPIC -shared $(PREDICT_OBJECTS) -o $(PREDICT_LIBRARY) $(PREDICT_LDFLAGS)

$(CRIMES_LIBRARY):  $(CRIMES_OBJECTS)
	$(NVCC)  --compiler-options -fPIC -shared $(CRIMES_OBJECTS) -o $(CRIMES_LIBRARY) $(CRIMES_LDFLAGS)

# ---- gcc C compile ------------------
predict.o:  predict.c predict.h
	$(CC) predict.c -c $(PREDICT_CFLAGS)

crimes.o:  crimes.cu crimes.h
	nvcc --compiler-options -fPIC  -c crimes.cu $(CRIMES_CFLAGS) $(GENCODE_FLAGS)

mgpucontext.o: moderngpu/src/mgpucontext.cu
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpucontext.cu $(CRIMES_CFLAGS) $(GENCODE_FLAGS) 

mgpuutil.o: moderngpu/src/mgpuutil.cpp
	nvcc --compiler-options -fPIC -c moderngpu/src/mgpuutil.cpp $(CRIMES_CFLAGS) $(GENCODE_FLAGS) 

clean:
	rm $(CRIMES_OBJECTS) $(PREDICT_OBJECTS)