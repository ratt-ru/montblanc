Montblanc
=========

A PyCUDA implementation of the Radio Interferometry Measurement
Equation, and a foothill of Mount Exaflop.

License
-------

Montblanc is licensed under the GNU GPL v2.0 License.

Requirements
------------

-  PyCUDA 2016.1
-  A Kepler NVIDIA GPU for more recent functionality.

Installation
------------

Certain pre-requisites must be installed, prior to calling the
``setup.py`` script.

Pre-requisites
~~~~~~~~~~~~~~

-  `libffi <https://sourceware.org/libffi/>`__ development files,
   required by cffi. On ubuntu 14.04, you can run:

   .. code:: bash

       $ sudo apt-get install libffi-dev

-  `casacore <https://github.com/casacore/casacore>`__ and the `measures <ftp://ftp.astron.nl/outgoing/Measures/>`__ found in casacore-data. Gijs Molenaar has kindly packaged this on Ubuntu/Debian style systems.

   On Ubuntu 14.04, these packages can be added from the `radio astronomy
   PPA <https://launchpad.net/~radio-astro/+archive/main>`__ :

   .. code:: bash

       $ sudo apt-get install software-properties-common
       $ sudo add-apt-repository ppa:radio-astro/main
       $ sudo apt-get update
       $ sudo apt-get install casacore21 casacore-data

   On Ubuntu 16.04 these packages can be added from the `kernsuite PPA
   <https://launchpad.net/~kernsuite/+archive/ubuntu/kern-1>`__:
 
   .. code:: bash
 
       $ sudo apt-get install software-properties-common
       $ sudo add-apt-repository ppa:kernsuite/kern-1
       $ sudo apt-get update
       $ sudo apt-get install casacore-dev casacore-data

   Otherwise, casacore and the measures tables will need to be manually installed.

Installing the package
~~~~~~~~~~~~~~~~~~~~~~

It is strongly recommended that you perform the install within a
`Virtual
Environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__.
If not, consider adding the ``--user`` flag to the following pip and
python commands to install within your home directory.

First install `numpy <http://www.numpy.org>`__ due to this
`issue <http://stackoverflow.com/questions/18997339/scipy-and-numpy-install-on-linux-without-root>`__.

::

    $ pip install numpy

Then, run

::

    $ python setup.py build

to build the package and

::

    $ python setup.py install

to install the package.

Possible Issues
~~~~~~~~~~~~~~~

-  `numexpr <https://github.com/pydata/numexpr>`__. When running
   ``python setup.py install``, if you see an error message like this:

   .. code:: python

       Traceback (most recent call last):
         File "setup.py", line 154, in <module>
           setup_package()
         File "setup.py", line 146, in setup_package
           from numpy.distutils.core import setup
       ImportError: No module named numpy.distutils.core

   Go back to `Installing the package <#installing-the-package>`__ and
   install `numpy <http://www.numpy.org>`__.

-  `PyCUDA 2016.1 <http://mathema.tician.de/software/pycuda/>`__.
   setup.py will attempt to install this automatically, but this might
   not work if you have a non-standard CUDA install location. It's worth
   running

   .. code:: bash

       $ python -c 'import pycuda.autoinit'

   to check if your pycuda can talk to the NVIDIA driver. If not,
   manually download and install
   `PyCUDA <http://mathema.tician.de/software/pycuda/>`__.

-  `cub 1.5.2 <https://github.com/nvlabs/cub>`__. setup.py will
   attempt to download this from github and install to the correct
   directory during install. If this fails do the following:

   .. code:: bash

       $ wget -c https://codeload.github.com/NVlabs/cub/zip/1.5.2
       $ mv 1.5.2 cub.zip
       $ python setup.py install

-  `python-casacore <https://github.com/casacore/python-casacore/>`__ is specified as a dependency in setup.py. If install fails here, you will need to manually install it and point it at your casacore libraries.

Running Tests
-------------

Once the libraries have been compiled you should be able to run the

::

    $ cd tests
    $ python -c 'import montblanc; montblanc.test()'
    $ python -m unittest test_rime_v4.TestRimeV4.test_sum_coherencies_double

which will run the entire test suite or only the specified test case,
respectively. The reported times are for the entire test case with numpy
code, and not just the CUDA kernels.

If you're running on an ubuntu laptop with optimus technology, you may
have to install bumblebee and run

::

    $ optirun python -c 'import montblanc; montblanc.test()'

Playing with a Measurement Set
------------------------------

You could also try run

::

    $ cd examples
    $ python MS_example.py /home/user/data/WSRT.MS -np 10 -ng 10 -c 100

which sets up things based on the supplied Measurement Set, with 10
point and 10 gaussian sources. It performs 100 iterations of the
pipeline.

Citing Montblanc
----------------

If you use Montblanc and find it useful, please consider citing the
related
`paper <http://www.sciencedirect.com/science/article/pii/S2213133715000633>`__.
A `arXiv <http://arxiv.org/abs/1501.07719>`__ preprint is available.

The BIRO paper is available at
`MNRAS <http://mnras.oxfordjournals.org/content/450/2/1308.abstract>`__,
and a `arXiv <http://arxiv.org/abs/1501.05304>`__ is also available.

Caveats
-------

Montblanc is an experimental package, undergoing rapid development. The
plan for 2015 is to iterate on new versions of the BIRO pipeline.

In general, I will avoid making changes to BIRO v2 and v3, but
everything beyond that may be changed, including the basic API residing
in BaseSolver.py. In practice, this means that the interfaces in the
base montblanc package will remain stable. For example:

.. code:: python

    import montblanc
    montblanc.rime_solver(...)

Everything should be considered unstable and subject to change. I will
make an effort to maintain the CHANGELOG.md, to record any breaking API
changes.
