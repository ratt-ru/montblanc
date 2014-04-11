# PipedRimes

A PyCUDA implementation of the Radio Interferometry Measurement Equation

## Installation

Pre-requisites must be installed and dependent C libraries built.

### Pre-requisites

You'll need to install the **python-pycuda** package on ubuntu

    # sudo apt-get install python-pycuda

### Setting up Submodules

You'll need to run

    # git submodule init
    # git submodule update

This should clone the **moderngpu** and **cub** CUDA libraries which are needed by pipedrimes.

### Building Libraries

Run **make** to compile the **predict.so** and **crimes.so** libraries.

You may need to configure the numpy include and library directories which are somewhat hardcoded at present. You also need to point your linker at the correct location of **libcuda.so**


## Running Tests

Once the libraries have been compiled you should be able to run

    # python TestRimes.py
    # python -m unittest TestRimes.TestRimes.test_predict_float

which will run the current test suite or only the particular test case specified. The reported times are for the entire test case with numpy code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may have to install bumblebee and run

    # optirun python TestRimes.py

## Playing with the Westerbork MeasurementSet

You could also try run

    # python MeasurementSetSharedData.py

which sets up things based on the Westerbork measurement set. However, a path to the measurement set must be correctly configured.
