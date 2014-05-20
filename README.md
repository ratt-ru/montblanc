# Montblanc

A PyCUDA implementation of the Radio Interferometry Measurement Equation, and a foothill of Mount Exaflop.

## Requirements

- PyCUDA 2013.1
- A Kepler NVIDIA GPU (probably)

## Installation

Pre-requisites must be installed and dependent C libraries built.

### Pre-requisites

You'll need to install the [PyCUDA][pycuda] package on ubuntu

    # sudo apt-get install python-pycuda

You'll also need to install the [pyrap][pyrap] library, which is dependent on [casacore][casacore]. It may be easier to add the [SKA PPA][ska-ppa]  and get the binaries from there.

### Setting up Submodules

You'll need to run

    # git submodule init
    # git submodule update

This should clone the [moderngpu][moderngpu] and [cub][cub] CUDA libraries which are needed by montblanc.

### Building the package

Run
     
    # python setup.py build

to build the package. This should automatically find your CUDA compiler and compile the necessary C extensions. The following:

    # python setup.py install

should install the package.

## Running Tests

Once the libraries have been compiled you should be able to run

    # python TestRimes.py
    # python -m unittest TestRimes.TestRimes.test_predict_float

which will run the current test suite or only the particular test case specified. The reported times are for the entire test case with numpy code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may have to install bumblebee and run

    # optirun python TestRimes.py

## Playing with the Westerbork MeasurementSet

You could also try run

    # python MS_example.py /home/user/data/WSRT.MS -n 17 -c 100

which sets up things based on the Westerbork Measurement Set, with 17 sources. It performs 100 iterations of the pipeline.

[pycuda]:http://mathema.tician.de/software/pycuda/
[moderngpu]:https://github.com/nvlabs/moderngpu
[cub]:https://github.com/nvlabs/cub
[pyrap]:https://code.google.com/p/pyrap/
[casacore]:https://code.google.com/p/casacore/
[ska-ppa]:https://launchpad.net/~ska-sa/+archive/main

