# Montblanc

A PyCUDA implementation of the Radio Interferometry Measurement Equation, and a foothill of Mount Exaflop.

## License

Montblanc is licensed under the GNU GPL v2.0 License.

## Requirements

- PyCUDA 2015.1.3
- A Kepler NVIDIA GPU for more recent functionality.

## Installation

Certain pre-requisites must be installed, prior to calling the `setup.py` script.

### Pre-requisites

- **[libffi][libffi]** development files, required by cffi. On ubuntu 14.04, you can run:

    ```bash
    $ sudo apt-get install libffi-dev
    ```
    
- **[python-casacore][pyrap]** which depends on **[casacore2][casacore]**.
    On Ubuntu 14.04 add the [radio astronomy PPA][radio-astro-ppa]  and get the binaries from there:

    ```bash
    $ sudo apt-get install software-properties-common
    $ sudo add-apt-repository ppa:radio-astro/main
    $ sudo apt-get update
    $ sudo apt-get install python-casacore
    ```

### Installing the package

It is strongly recommended that you perform the install within a [Virtual Environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
If not, consider adding the `--user` flag to the following
pip and python commands to install within your home directory.

First install [numpy] due to this [issue](http://stackoverflow.com/questions/18997339/scipy-and-numpy-install-on-linux-without-root).

    $ pip install numpy

Then, run

    $ python setup.py build

to build the package and

    $ python setup.py install

to install the package.

### Possible Issues

- **[PyCUDA 2015.1.3][pycuda]**. setup.py will attempt to install this automatically,
    but this might not work if you have a non-standard CUDA install location. It's worth running

    ```bash
    $ python -c 'import pycuda.autoinit'
    ```
    
    to check if your pycuda can talk to the NVIDIA driver. If not, manually download
    and install [PyCUDA][pycuda].
- **[cub 1.5.1][cub]**. setup.py will attempt to download this from github
    and install to the correct directory during install. If this fails do the following:

    ```bash
    $ wget -c https://codeload.github.com/NVlabs/cub/zip/1.5.1
    $ mv 1.5.1 cub.zip
    $ python setup.py install
    ```

## Running Tests

Once the libraries have been compiled you should be able to run the

    $ cd tests
    $ python -c 'import montblanc; montblanc.test()'
    $ python -m unittest test_biro_v4.TestBiroV4.test_sum_coherencies_double

which will run the entire test suite or only the specified test case, respectively.
The reported times are for the entire test case with numpy code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may have to install bumblebee and run

    $ optirun python -c 'import montblanc; montblanc.test()'

## Playing with a Measurement Set

You could also try run

    $ cd examples
    $ python MS_example.py /home/user/data/WSRT.MS -np 10 -ng 10 -c 100

which sets up things based on the supplied Measurement Set, with 10 point and 10 gaussian sources. It performs 100 iterations of the pipeline.

## Citing Montblanc

If you use Montblanc and find it useful, please consider
citing the related [paper][montblanc-paper-sciencedirect].
A [arXiv][montblanc-paper-arxiv] preprint is available.

The BIRO paper is available at [MNRAS][biro-paper-mnras],
and a [arXiv][biro-paper-arxiv] is also available.

## Caveats

Montblanc is an experimental package, undergoing rapid development.
The plan for 2015 is to iterate on new versions of the BIRO pipeline.

In general, I will avoid making changes to BIRO v2 and v3,
but everything beyond that may be changed, including the basic API residing in BaseSolver.py.
In practice, this means that the interfaces in the base montblanc package will remain stable.
For example:

```python
import montblanc
montblanc.rime_solver(...)
```

Everything should be considered unstable and subject to change.
I will make an effort to maintain the CHANGELOG.md, to record any breaking API changes.

[pycuda]:http://mathema.tician.de/software/pycuda/
[pytools]:https://pypi.python.org/pypi/pytools
[moderngpu]:https://github.com/nvlabs/moderngpu
[cub]:https://github.com/nvlabs/cub
[pyrap]:https://github.com/casacore/python-casacore
[casacore]:https://github.com/casacore/casacore
[radio-astro-ppa]:https://launchpad.net/~radio-astro/+archive/main
[libffi]:https://sourceware.org/libffi/
[biro-paper-arxiv]:http://arxiv.org/abs/1501.05304
[biro-paper-mnras]:http://mnras.oxfordjournals.org/content/450/2/1308.abstract
[montblanc-paper-arxiv]:http://arxiv.org/abs/1501.07719
[montblanc-paper-sciencedirect]:http://www.sciencedirect.com/science/article/pii/S2213133715000633
[numpy]:http://www.numpy.org
