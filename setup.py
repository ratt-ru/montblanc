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
import sys

# =============
# Setup logging
# =============

from install.install_log import log
from install.cuda import inspect_cuda, InspectCudaException
from install.cub import install_cub, InstallCubException


# ==================
# setuptools imports
# ==================

from distutils.version import LooseVersion
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution

import versioneer

PY2 = sys.version_info[0] == 2

mb_path = 'montblanc'
mb_inc_path = os.path.join(mb_path, 'include')

# =================
# Detect readthedocs
# ==================

on_rtd = os.environ.get('READTHEDOCS') == 'True'



REQ_TF_VERSION = LooseVersion("1.8.0")

# Inspect previous tensorflow installs
try:
    import tensorflow as tf
except ImportError:
    if not on_rtd:
        raise ImportError("Please 'pip install tensorflow==%s' or "
                          "'pip install tensorflow-gpu==%s' prior to "
                          "installation if you require CPU or GPU "
                          "support, respectively" %
                          (REQ_TF_VERSION, REQ_TF_VERSION))

    tf_installed = False
    use_tf_cuda = False
else:
    found_version = LooseVersion(tf.__version__)

    if found_version < REQ_TF_VERSION:
        raise ValueError("Installed version of tensorflow is %s "
                         "but %s is required" %
                         (found_version, REQ_TF_VERSION))

    tf_installed = True
    use_tf_cuda = tf.test.is_built_with_cuda()

# ===========================
# Detect CUDA and GPU Devices
# ===========================

# See if CUDA is installed and if any NVIDIA devices are available
# Choose the tensorflow flavour to install (CPU or GPU)
from install.cuda import inspect_cuda, InspectCudaException
from install.cub import install_cub, InstallCubException

if use_tf_cuda:
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
        log.exception("CUDA not found: {}. ".format(str(e)))
        raise

    except InstallCubException as e:
        # This shouldn't happen and the user should
        # fix it based on the exception
        log.exception("NVIDIA cub install failed.")
        raise
else:
    device_info, nvcc_settings = {}, {'cuda_available': False}


def readme():
    """ Return README.rst contents """
    with open('README.rst') as f:
        return f.read()


install_requires = [
    'attrdict >= 2.0.0',
    'attrs >= 16.3.0',
    'enum34 >= 1.1.6; python_version <= "2.7"',
    'funcsigs >= 0.4',
    'futures >= 3.0.5; python_version <= "2.7"',
    'six',
    'hypercube == 0.3.4',
    'tensorflow == {0:s}'.format(str(REQ_TF_VERSION)),
]

# ==================================
# Avoid binary packages and compiles
# on readthedocs
# ==================================

if on_rtd:
    cmdclass = {}
    ext_modules = []
    ext_options = {}
else:
    # Add binary/C extension type packages
    install_requires += [
        'astropy >= 2.0.0, < 3.0; python_version <= "2.7"',
        'astropy > 3.0; python_version >= "3.0"',
        'cerberus >= 1.1',
        'nose >= 1.3.7',
        'numba >= 0.36.2',
        'numpy >= 1.11.3',
        'python-casacore >= 2.1.2',
        'ruamel.yaml >= 0.15.22',
    ]

    if not tf_installed:
        log.info("No previous version of tensorflow discovered. "
                 "The CPU version will be installed.")

        install_requires.append("tensorflow == %s" % REQ_TF_VERSION)

    from install.tensorflow_ops_ext import (BuildCommand,
                                            tensorflow_extension_name,
                                            create_tensorflow_extension)

    cmdclass = {'build_ext': BuildCommand}
    # tensorflow_ops_ext.BuildCommand.run will
    # expand this dummy extension to its full portential

    ext_modules = [create_tensorflow_extension(nvcc_settings, device_info)]

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
