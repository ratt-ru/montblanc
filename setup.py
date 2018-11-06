#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Simon Perkins
#
# This file is part of montblanc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import json
import os
from os.path import join as pjoin
import sys

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution

import versioneer

from install.install_log import log
from install.cuda import inspect_cuda, InspectCudaException
from install.cub import install_cub, InstallCubException


# ==============
# Setup logging
# ==============


mb_path = 'montblanc'
mb_inc_path = pjoin(mb_path, 'include')

# ===================
# Detect readthedocs
# ====================

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# =======================
# Monkeypatch distutils
# =======================

# Save the original command for use within the monkey-patched version
_DISTUTILS_REINIT = Distribution.reinitialize_command


def reinitialize_command(self, command, reinit_subcommands):
    """
    Monkeypatch distutils.Distribution.reinitialize_command() to match behavior
    of Distribution.get_command_obj()
    This fixes a problem where 'pip install -e' does not reinitialise options
    using the setup(options={...}) variable for the build_ext command.
    This also effects other option sourcs such as setup.cfg.
    """
    cmd_obj = _DISTUTILS_REINIT(self, command, reinit_subcommands)

    options = self.command_options.get(command)

    if options:
        self._set_command_options(cmd_obj, options)

    return cmd_obj


# Replace original command with monkey-patched version
Distribution.reinitialize_command = reinitialize_command

TF_VERSION = "1.12.0"
TF_INSTALL_MSG = ("Please 'pip install tensorflow==%s' or "
                  "'pip install tensorflow-gpu==%s' prior to "
                  "installation if you require CPU or GPU "
                  "support, respectively" % (TF_VERSION, TF_VERSION))

try:
    import tensorflow as tf
except ImportError as e:
    ex = ImportError("Tensorflow import failed: %s\n" + TF_INSTALL_MSG)
    raise (ex, None, sys.exc_info()[2])

else:
    if not tf.__version__ == TF_VERSION:
        raise ValueError(("Tensorflow '%s' is required, "
                          "but '%s' was found\n" + TF_INSTALL_MSG)
                         % (TF_VERSION, tf.__version__))

    use_tf_cuda = tf.test.is_built_with_cuda()


# ============================
# Detect CUDA and GPU Devices
# ============================

# See if CUDA is installed and if any NVIDIA devices are available
# Choose the tensorflow flavour to install (CPU or GPU)

try:
    # Look for CUDA devices and NVCC/CUDA installation
    device_info, nvcc_settings = inspect_cuda()
    tensorflow_package = 'tensorflow-gpu'

    cuda_version = device_info['cuda_version']
    log.info("CUDA '{}' found. "
             "Installing tensorflow GPU".format(cuda_version))

    log.info("CUDA installation settings:\n{}"
             .format(json.dumps(nvcc_settings, indent=2)))

    log.info("CUDA code will be compiled for the following devices:\n{}"
             .format(json.dumps(device_info['devices'], indent=2)))

    # Download and install cub
    install_cub(mb_inc_path)

except InspectCudaException as e:
    # Can't find a reasonable NVCC/CUDA install. Go with the CPU version
    log.exception("CUDA not found")
    raise

except InstallCubException as e:
    # This shouldn't happen and the user should fix it based on the exception
    log.exception("NVIDIA cub install failed.")
    raise


def readme():
    """ Return README.rst contents """
    with open('README.rst') as f:
        return f.read()


install_requires = [
    'attrdict >= 2.0.0',
    'attrs >= 16.3.0',
    'bitstring >= 3.1.5',
    'boltons >= 17.1.0',
    'dask >= 0.17.1',
    'distributed >= 1.21.3',
    'futures >= 3.0.5',
    'hypercube == 0.3.3',
    'xarray-ms >= 0.0.1',
]

# ===================================
# Avoid binary packages and compiles
# on readthedocs
# ===================================

if on_rtd:
    cmdclass = {}
    ext_modules = []
    ext_options = {}
else:
    # Add binary/C extension type packages
    install_requires += [
        'astropy >= 1.3.0',
        'cerberus >= 1.1',
        'cppimport >= 17.9.18',
        'numpy >= 1.11.3',
        'pybind11 >= 2.2.0',
        'python-casacore >= 2.1.2',
        'ruamel.yaml >= 0.15.22',
    ]

    from install.tensorflow_ops_ext import (BuildCommand,
                                            tensorflow_extension_name)

    cmdclass = {'build_ext': BuildCommand}
    # tensorflow_ops_ext.BuildCommand.run will
    # expand this dummy extension to its full portential
    ext_modules = [Extension(tensorflow_extension_name, ['rime.cu'])]
    # Pass NVCC and CUDA settings through to the build extension
    ext_options = {
        'build_ext': {
            'nvcc_settings': nvcc_settings,
            'cuda_devices': device_info,
        },
    }

log.info('install_requires={}'.format(install_requires))

setup(name='montblanc',
      version=versioneer.get_version(),
      description='GPU-accelerated RIME implementations.',
      long_description=readme(),
      url='http://github.com/ska-sa/montblanc',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "License :: Other/Proprietary License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "Topic :: Scientific/Engineering :: Astronomy",
      ],
      author='Simon Perkins',
      author_email='simon.perkins@gmail.com',
      cmdclass=versioneer.get_cmdclass(cmdclass),
      ext_modules=ext_modules,
      options=ext_options,
      license='GPL2',
      install_requires=install_requires,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
