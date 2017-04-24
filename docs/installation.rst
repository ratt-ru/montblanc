Requirements
------------

- CUDA 8.0
- A Kepler NVIDIA GPU.

Installation
------------

Certain pre-requisites must be installed:

Pre-requisites
~~~~~~~~~~~~~~

- CUDA_ 8.0. It could be easier install from the NVIDIA site on Linux systems.

-  casacore_ and the `measures <ftp://ftp.astron.nl/outgoing/Measures/>`__ found in casacore-data. Gijs Molenaar has kindly packaged this on Ubuntu/Debian style systems.

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

- Check that the python-casacore_ and casacore_ dependencies are installed. By default python-casacore_ builds from pip and therefore from source. To succeed, library dependencies such as `libboost-python` must be installed beforehand. Additionally, python-casacore depends on casacore. Even though kernsuite installs casacore, it may not install the development package dependencies (headers) that python-casacore needs to compile.

Installing the package
~~~~~~~~~~~~~~~~~~~~~~

Set the `CUDA_PATH` so that the setup script can find CUDA:

.. code:: bash

    $ export CUDA_PATH=/usr/local/cuda-8.0

If `nvcc` is installed in `/usr/bin/nvcc` (as in a standard Ubuntu installation)
or somewhere on your `PATH`, you can leave `CUDA_PATH` unset. In this case
setup will infer the CUDA_PATH as `/usr`


It is strongly recommended that you perform the install within a
`Virtual Environment <venv>`_.
If not, consider adding the ``--user`` flag to the following pip and
python commands to install within your home directory.

.. code:: bash

    $ virtualenv $HOME/mb
    $ source virtualenv $HOME/mb/bin/activate
    (mb) $ pip install -U pip setuptools wheel


Then, run:

.. code:: bash

    (mb) $ pip install git+git://github.com/ska-sa/montblanc.git@rime-tf

Installing the package in development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository, checkout the rime-tf branch
and pip install montblanc in development mode.

.. code:: bash

    (mb) $ git clone --branch rime-tf git://github.com/ska-sa/montblanc.git
    (mb) $ pip install -e $HOME/montblanc

Possible Issues
~~~~~~~~~~~~~~~

-  `cub 1.6.4 <cub>`_. The setup script will
   attempt to download this from github and install to the correct
   directory during install. If this fails do the following:

   .. code:: bash

       $ wget -c https://codeload.github.com/NVlabs/cub/zip/1.6.4
       $ mv 1.6.4 cub.zip
       $ pip install -e .

-  `python-casacore`_ is
   specified as a dependency in setup.py. If installation fails here, you will
   need to manually install it and point it at your casacore libraries.

.. _cuda: https://developer.nvidia.com/cuda-downloads
.. _cub: https://github.com/nvlabs/cub
.. _casacore: https://github.com/casacore/casacore
.. _python-casacore: https://github.com/casacore/python-casacore
.. _venv: http://docs.python-guide.org/en/latest/dev/virtualenvs/
