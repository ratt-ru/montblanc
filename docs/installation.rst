Requirements
------------

If you wish to take advantage of GPU Acceleration, the following are required:

- `CUDA 8.0  <CUDA_>`_.
- `cuDNN 6.0 <cudnn_>`_ for CUDA 8.0.
- A Kepler or later model NVIDIA GPU.

Installation
------------

Certain pre-requisites must be installed:

Pre-requisites
~~~~~~~~~~~~~~

- .. _install_tf_gpu:

  Montblanc depends on tensorflow_ for CPU and GPU acceleration.
  By default the CPU version of tensorflow is installed during
  Montblanc's installation process.
  If you require GPU acceleration, the GPU version of tensorflow
  should be installed first.

  .. code:: bash

    $ pip install tensorflow-gpu==1.8.0

- GPU Acceleration requires `CUDA 8.0 <CUDA_>`_ and `cuDNN 6.0 for CUDA 8.0 <cudnn_>`_.

  - It is often easier to CUDA install from the `NVIDIA <CUDA_>`_ site on Linux systems.
  - You will need to sign up for the `NVIDIA Developer Program <cudnn_>`_ to download cudNN.

  During the installation process, Montblanc will inspect your CUDA installation
  to determine if a GPU-supported installation can proceed.
  If your CUDA installation does not live in ``/usr``, it  helps to set a
  number of environment variables for this to proceed smoothly.
  **For example**, if CUDA is installed in ``/usr/local/cuda-8.0`` and cuDNN is unzipped
  into ``/usr/local/cudnn-6.0-cuda-8.0``, run the following on the command line or
  place it in your ``.bashrc``

  .. code:: bash

      # CUDA 8
      $ export CUDA_PATH=/usr/local/cuda-8.0
      $ export PATH=$CUDA_PATH/bin:$PATH
      $ export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
      $ export LD_LIBRARY_PATH=$CUDA_PATH/extras/CUPTI/lib64/:$LD_LIBRARY_PATH

      # CUDNN 6.0 (CUDA 8.0)
      $ export CUDNN_HOME=/usr/local/cudnn-6.0-cuda-8.0
      $ export C_INCLUDE_PATH=$CUDNN_HOME/include:$C_INCLUDE_PATH
      $ export CPLUS_INCLUDE_PATH=$CUDNN_HOME/include:$CPLUS_INCLUDE_PATH
      $ export LD_LIBRARY_PATH=$CUDNN_HOME/lib64:$LD_LIBRARY_PATH

      # Latest NVIDIA drivers
      $ export LD_LIBRARY_PATH=/usr/lib/nvidia-375:$LD_LIBRARY_PATH

-  casacore_ and the `measures <ftp://ftp.astron.nl/outgoing/Measures/>`__ found in casacore-data.
   Gijs Molenaar has kindly packaged this as kernsuite_ on as Ubuntu/Debian style systems.


   Otherwise, casacore and the measures tables should be manually installed.

- .. _check_dependencies:

  Check that the python-casacore_ and
  casacore_ _`dependencies are installed`.
  By default python-casacore_ builds from pip and therefore from source.
  To succeed, library dependencies such as ``libboost-python`` must be installed beforehand.
  Additionally, python-casacore depends on casacore.
  Even though kernsuite installs casacore, it may not install the development
  package dependencies (headers) that python-casacore needs to compile.

Installing the package
~~~~~~~~~~~~~~~~~~~~~~

Set the ``CUDA_PATH`` so that the setup script can find CUDA:

.. code:: bash

    $ export CUDA_PATH=/usr/local/cuda-8.0

If ``nvcc`` is installed in ``/usr/bin/nvcc`` (as in a standard Ubuntu installation)
or somewhere on your ``PATH``, you can leave ``CUDA_PATH`` unset. In this case
setup will infer the CUDA_PATH as ``/usr``

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

    (mb) $ pip install --log=mb.log git+git://github.com/ska-sa/montblanc.git@master

Installing the package in development mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository, checkout the master branch
and pip install montblanc in development mode.

.. code:: bash

    (mb) $ git clone git://github.com/ska-sa/montblanc.git
    (mb) $ pip install --log=mb.log -e $HOME/montblanc

Possible Issues
~~~~~~~~~~~~~~~

- Montblanc doesn't use your GPU or compile GPU tensorflow operators.

  1. Check if the `GPU version of tensorflow <install_tf_gpu_>`_ is installed.

     It is possible to see if the GPU version of tensorflow is installed by running
     the following code in a python interpreter:

     .. code:: python

          import tensorflow as tf
          with tf.Session() as S: pass

     If tensorflow knows about your GPU it will log some information about it:

     .. code:: bash

          2017-05-16 14:24:38.571320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties:
          name: GeForce GTX 960M
          major: 5 minor: 0 memoryClockRate (GHz) 1.176
          pciBusID 0000:01:00.0
          Total memory: 3.95GiB
          Free memory: 3.92GiB
          2017-05-16 14:24:38.571352: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0
          2017-05-16 14:24:38.571372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y
          2017-05-16 14:24:38.571403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 960M, pci bus id: 0000:01:00.0)

  2. The installation process couldn't find your CUDA install.

     It will log information about where it thinks this is and which GPU devices
     you have installed.

     Check the install log generated by the ``pip`` commands given above to see
     why this fails, searching for "**Montblanc Install**" entries.

-  `cub 1.6.4 <cub>`_. The setup script will
   attempt to download this from github and install to the correct
   directory during install. If this fails do the following:

   .. code:: bash

       $ wget -c https://codeload.github.com/NVlabs/cub/zip/1.6.4
       $ mv 1.6.4 cub.zip
       $ pip install -e .

-  `python-casacore`_ is
   specified as a dependency in setup.py. If installation fails here:

    1. Check that the `python-casacore dependencies <check_dependencies_>`_ are installed.
    2. You will need to manually install it and point it at your casacore libraries.

.. _cuda: https://developer.nvidia.com/cuda-downloads
.. _cudnn: https://developer.nvidia.com/cudnn
.. _cub: https://github.com/nvlabs/cub
.. _casacore: https://github.com/casacore/casacore
.. _kernsuite: http://kernsuite.info/
.. _python-casacore: https://github.com/casacore/python-casacore
.. _venv: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _tensorflow: https://tensorflow.org/
.. _tensorflow-gpu: https://pypi.python.org/pypi/tensorflow-gpu
.. _tensorflow-cpu: https://pypi.python.org/pypi/tensorflow
