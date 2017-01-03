Montblanc
=========

A PyCUDA implementation of the Radio Interferometry Measurement
Equation, and a foothill of Mount Exaflop.

License
-------

Montblanc is licensed under the GNU GPL v2.0 License.

Requirements
------------

- CUDA 7.5
- A Kepler NVIDIA GPU.

Installation
------------

Certain pre-requisites must be installed:

Pre-requisites
~~~~~~~~~~~~~~

-  `casacore <https://github.com/casacore/casacore>`__ and the `measures <ftp://ftp.astron.nl/outgoing/Measures/>`__ found in casacore-data. Gijs Molenaar has kindly packaged this on Ubuntu/Debian style systems.

   On Ubuntu 14.04, these packages can be added from the `radio astronomy
   PPA <https://launchpad.net/~radio-astro/+archive/main>`__ :

   .. code:: bash

       $ sudo apt-get install software-properties-common
       $ sudo add-apt-repository ppa:radio-astro/main
       $ sudo apt-get update
       $ sudo apt-get install libcasacore2-dev casacore-data

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

Set the `CUDA_PATH` so that the setup script can find CUDA:

::

    $ export CUDA_PATH=/usr/local/cuda-7.5

If `nvcc` is installed in `/usr/bin/nvcc` (as in a standard Ubuntu installation)
or somewhere on your `PATH`, you can leave `CUDA_PATH` unset. In this case
setup will infer the CUDA_PATH as `/usr/bin/..`


It is strongly recommended that you perform the install within a
`Virtual
Environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`__.
If not, consider adding the ``--user`` flag to the following pip and
python commands to install within your home directory.

::

    $ virtualenv $HOME/mb
    $ source virtualenv $HOME/mb/bin/activate
    (mb) $ pip install -U pip setuptools wheel


Then, run:

::

    (mb) $ pip install git+git://github.com/ska-sa/montblanc.git@rime-tf

Installing the package in development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository, checkout the rime-tf branch
and pip install montblanc in development mode.

::

    (mb) $ git clone --branch rime-tf git://github.com/ska-sa/montblanc.git
    (mb) $ pip install -e $HOME/montblanc

Possible Issues
~~~~~~~~~~~~~~~

-  `cub 1.6.4 <https://github.com/nvlabs/cub>`__. The setup script will
   attempt to download this from github and install to the correct
   directory during install. If this fails do the following:

   .. code:: bash

       $ wget -c https://codeload.github.com/NVlabs/cub/zip/1.6.4
       $ mv 1.6.4 cub.zip
       $ pip install -e .

-  `python-casacore <https://github.com/casacore/python-casacore/>`__ is
   specified as a dependency in setup.py. If installation fails here, you will
   need to manually install it and point it at your casacore libraries.

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
