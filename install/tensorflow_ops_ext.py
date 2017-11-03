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

import inspect
import itertools
import os

from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

from install_log import log

tensorflow_extension_name = 'montblanc.extensions.tensorflow.rime'

def customize_compiler_for_nvcc(compiler, nvcc_settings, device_info):
    """inject deep into distutils to customize gcc/nvcc dispatch """

    # tell the compiler it can process .cu files
    compiler.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = compiler.compiler_so
    default_compile = compiler._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            compiler.set_executable('compiler_so', nvcc_settings['nvcc_path'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        default_compile(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        compiler.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    compiler._compile = _compile


def cuda_architecture_flags(device_info):
    """
    Emit a list of architecture flags for each CUDA device found
    ['--gpu-architecture=sm_30', '--gpu-architecture=sm_52']
    """
    # Figure out the necessary device architectures
    if len(device_info['devices']) == 0:
        archs = ['--gpu-architecture=sm_30']
        log.info("No CUDA devices found, defaulting to architecture '{}'".format(archs[0]))
    else:
        archs = set()

        for device in device_info['devices']:
            arch_str = '--gpu-architecture=sm_{}{}'.format(device['major'], device['minor'])
            log.info("Using '{}' for '{}'".format(arch_str, device['name']))
            archs.add(arch_str)

    return list(archs)

def create_tensorflow_extension(nvcc_settings, device_info):
    """ Create an extension that builds the custom tensorflow ops """
    import tensorflow as tf
    import glob

    use_cuda = (bool(nvcc_settings['cuda_available'])
        and tf.test.is_built_with_cuda())

    # Source and includes
    source_path = os.path.join('montblanc', 'impl', 'rime', 'tensorflow', 'rime_ops')
    sources = glob.glob(os.path.join(source_path, '*.cpp'))

    # Header dependencies
    depends = glob.glob(os.path.join(source_path, '*.h'))

    # Include directories
    tf_inc = tf.sysconfig.get_include()
    include_dirs = [os.path.join('montblanc', 'include'), source_path]
    include_dirs += [tf_inc, os.path.join(tf_inc, "external", "nsync", "public")]

    # Libraries
    library_dirs = [tf.sysconfig.get_lib()]
    libraries = ['tensorflow_framework']
    extra_link_args = ['-fPIC', '-fopenmp', '-g0']

    # Macros
    define_macros = [
        ('_MWAITXINTRIN_H_INCLUDED', None),
        ('_FORCE_INLINES', None),
        ('_GLIBCXX_USE_CXX11_ABI', 0)]

    # Common flags
    flags = ['-std=c++11']

    gcc_flags = flags + ['-g0', '-fPIC', '-fopenmp', '-O2']
    gcc_flags += ['-march=native', '-mtune=native']
    nvcc_flags = flags + []

    # Add cuda specific build information, if it is available
    if use_cuda:
        # CUDA source files
        sources += glob.glob(os.path.join(source_path, '*.cu'))
        # CUDA include directories
        include_dirs += nvcc_settings['include_dirs']
        # CUDA header dependencies
        depends += glob.glob(os.path.join(source_path, '*.cuh'))
        # CUDA libraries
        library_dirs += nvcc_settings['library_dirs']
        libraries += nvcc_settings['libraries']
        # Flags
        nvcc_flags += ['-x', 'cu']
        nvcc_flags += ['--compiler-options', '"-fPIC"']
        # --gpu-architecture=sm_xy flags
        nvcc_flags += cuda_architecture_flags(device_info)
        # Ideally this would be set in define_macros, but
        # this must be set differently for gcc and nvcc
        nvcc_flags += ['-DGOOGLE_CUDA=%d' % int(use_cuda)]

    return Extension(tensorflow_extension_name,
        sources=sources,
        include_dirs=include_dirs,
        depends=depends,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler_for_nvcc() above
        extra_compile_args={ 'gcc': gcc_flags, 'nvcc': nvcc_flags },
        extra_link_args=extra_link_args,
    )


class BuildCommand(build_ext):
    """ Custom build command for building the tensorflow extension """
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.nvcc_settings = None
        self.cuda_devices = None

    def finalize_options(self):
        build_ext.finalize_options(self)

    def run(self):
        # Create the tensorflow extension during the run
        # At this point, pip should have installed tensorflow
        ext = create_tensorflow_extension(self.nvcc_settings,
            self.cuda_devices)

        for i, e in enumerate(self.extensions):
            if not e.name == ext.name:
                continue

            # Copy extension attributes over to the dummy extension.
            # Need to do this because the dummy extension has extra attributes
            # created on it during finalize_options() that are required by run()
            # and build_extensions(). However, tensorflow will not yet be installed
            # at this point
            for n, v in inspect.getmembers(ext):
                setattr(e, n, v)

        build_ext.run(self)

    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler,
            self.nvcc_settings, self.cuda_devices)
        build_ext.build_extensions(self)
