# Montblanc

A PyCUDA implementation of the Radio Interferometry Measurement Equation, and a foothill of Mount Exaflop.

## License

Montblanc is licensed under the GNU GPL v2.0 License.

## Requirements

- PyCUDA 2015.1
- A Kepler NVIDIA GPU for more recent functionality.

## Installation

Certain pre-requisites must be installed, prior to calling the `setup.py` script.

### Pre-requisites

Montblanc depends on [NVIDIA cub 1.4.1][cub]. It can be cloned as a submodule
prior to running `python setup.py`:

    # git submodule init
    # git submodule update --depth 1

An experimental alternative is to download [NVIDIA cub 1.4.1][cub] and unzip it
in the include directory of `nvcc`.

You'll also need to install the [python-casacore][pyrap] library which, in turn, is dependent on [casacore2][casacore]. It may be easier to add the [radio astronomy PPA][radio-astro-ppa]  and get the binaries from there.

### Building the package

Run

    # python setup.py build

to build the package. The following:

    # python setup.py install

should install the package.

## Running Tests

Once the libraries have been compiled you should be able to run the

    # cd tests
    # python -c 'import montblanc; montblanc.test()'
    # python -m unittest test_biro_v4.TestBiroV2.test_biro_v4

which will run the current test suite or only the particular test case specified. The reported times are for the entire test case with numpy code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may have to install bumblebee and run

    # optirun python -c 'import montblanc; montblanc.test()'

## Playing with a Measurement Set

You could also try run

    # cd examples
    # python MS_example.py /home/user/data/WSRT.MS -np 10 -ng 10 -c 100

which sets up things based on the supplied Measurement Set, with 10 point and 10 gaussian sources. It performs 100 iterations of the pipeline.

## Citing Montblanc

If you use Montblanc and find it useful, you may wish to consider citing the related paper. It is currently undergoing review, but an [arXiv][montblanc] preprint is available.

More information on BIRO can be found in this [arXiv][biro] preprint.

## Caveats

Montblanc is an experimental package, undergoing rapid development. The plan for 2015 is to iterate on new versions of the BIRO pipeline.

In general, I will avoid making changes to BIRO v2 and v3, but everything beyond that may be changed, including the basic API residing in BaseSolver.py. In practice, this means that the interfaces in the base montblanc package will remain stable. For example:

```python
import montblanc
montblanc.get_biro_solver(...)
```

Everything should be considered unstable and subject to change. I will make an effort to maintain the CHANGELOG.md, to record any breaking API changes.

[pycuda]:http://mathema.tician.de/software/pycuda/
[pytools]:https://pypi.python.org/pypi/pytools
[moderngpu]:https://github.com/nvlabs/moderngpu
[cub]:https://github.com/nvlabs/cub
[pyrap]:https://github.com/casacore/python-casacore
[casacore]:https://github.com/casacore/casacore
[radio-astro-ppa]:https://launchpad.net/~radio-astro/+archive/main
[biro]:http://arxiv.org/abs/1501.05304
[montblanc]:http://arxiv.org/abs/1501.07719
