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

from distutils import ccompiler
from distutils import sysconfig
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import json
import glob
import os
import shutil
import subprocess
import sys
import tempfile

from install_log import log

minimum_cuda_version = 6050

def find_in_path(filename, paths):
    for p in paths:
        fp = os.path.join(p, filename)
        if os.path.exists(fp):
            return os.path.abspath(fp)

    return os.path.join('usr', 'bin', filename)

class InspectCudaException(Exception):
    pass

def nvcc_compiler_settings():
    """ Find nvcc and the CUDA installation """

    search_paths = os.environ.get('PATH', '').split(os.pathsep)
    nvcc_path = find_in_path('nvcc', search_paths)
    default_cuda_path = os.path.join('usr', 'local', 'cuda')
    cuda_path = os.environ.get('CUDA_PATH', default_cuda_path)

    nvcc_found = os.path.exists(nvcc_path)
    cuda_path_found = os.path.exists(cuda_path)

    # Can't find either NVCC or some CUDA_PATH
    if not nvcc_found and not cuda_path_found:
        raise InspectCudaException("Neither nvcc '{}' "
            "or the CUDA_PATH '{}' were found!".format(
                nvcc_path, cuda_path))

    # No NVCC, try find it in the CUDA_PATH
    if not nvcc_found:
        log.warn("nvcc compiler not found at '{}'. "
            "Searching within the CUDA_PATH '{}'"
                .format(nvcc_path, cuda_path))

        bin_dir = os.path.join(cuda_path, 'bin')
        nvcc_path = find_in_path('nvcc', bin_dir)
        nvcc_found = os.path.exists(nvcc_path)

        if not nvcc_found:
            raise InspectCudaException("nvcc not found in '{}' "
                "or under the CUDA_PATH at '{}' "
                .format(search_paths, cuda_path))

    # No CUDA_PATH found, infer it from NVCC
    if not cuda_path_found:
        cuda_path = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), ".."))

        log.warn("CUDA_PATH not found, inferring it as '{}' "
            "from the nvcc location '{}'".format(
                cuda_path, nvcc_path))

        cuda_path_found = True

    # Set up the compiler settings
    include_dirs = []
    library_dirs = []
    define_macros = []

    if cuda_path_found:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if sys.platform == 'win32':
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if sys.platform == 'darwin':
        library_dirs.append(os.path.join(default_cuda_path, 'lib'))

    return {
        'cuda_available' : True,
        'nvcc_path' : nvcc_path,
        'include_dirs': include_dirs,
        'library_dirs': library_dirs,
        'define_macros': define_macros,
        'libraries' : ['cudart', 'cuda'],
        'language': 'c++',
    }

def inspect_cuda_version_and_devices(compiler, settings):
    """
    Poor mans deviceQuery. Returns CUDA_VERSION information and
    CUDA device information in JSON format
    """
    try:
        output = build_and_run(compiler, '''
            #include <cuda.h>
            #include <stdio.h>

            __device__ void test(int * in, int * out)
            {
                int tid = blockIdx.x*blockDim.x + threadIdx.x;
                out[tid] = in[tid];
            }

            int main(int argc, char* argv[]) {

              printf("{\\n");
              printf("  \\"cuda_version\\": %d,\\n", CUDA_VERSION);

              printf("  \\"devices\\": [\\n");

              int nr_of_devices = 0;
              cudaGetDeviceCount(&nr_of_devices);

              for(int d=0; d < nr_of_devices; ++d)
              {
                cudaDeviceProp p;
                cudaGetDeviceProperties(&p, d);

                printf("    {\\n");

                bool last = (d == nr_of_devices-1);

                printf("      \\"name\\": \\"%s\\",\\n", p.name);
                printf("      \\"major\\": %d,\\n", p.major);
                printf("      \\"minor\\": %d,\\n", p.minor);
                printf("      \\"memory\\": %lu\\n", p.totalGlobalMem);

                printf("    }%s\\n", last ? "" : ",");
              }

              printf("  ]\\n");
              printf("}\\n");

              return 0;
            }
        ''',
        filename='test.cu',
        include_dirs=settings['include_dirs'],
        library_dirs=settings['library_dirs'],
        libraries=settings['libraries'])

    except Exception as e:
        msg = ("Running the CUDA device check "
            "stub failed\n{}".format(str(e)))
        raise InspectCudaException(msg), None, sys.exc_info()[2]

    return output

def build_and_run(compiler, source, filename, libraries=(),
                  include_dirs=(), library_dirs=()):
    temp_dir = tempfile.mkdtemp()

    try:
        fname = os.path.join(temp_dir, filename)
        with open(fname, 'w') as f:
            f.write(source)

        objects = compiler.compile([fname], output_dir=temp_dir,
                                   include_dirs=include_dirs)

        try:
            postargs = ['/MANIFEST'] if sys.platform == 'win32' else []
            compiler.link_executable(objects,
                                     os.path.join(temp_dir, 'a'),
                                     libraries=libraries,
                                     library_dirs=library_dirs,
                                     extra_postargs=postargs,
                                     target_lang='c++')
        except Exception as e:
            msg = ('Cannot build a stub file.\n'
                'Original error: {0}'.format(e))
            raise InspectCudaException(msg)

        try:
            out = subprocess.check_output(os.path.join(temp_dir, 'a'))
            return out

        except Exception as e:
            msg = ('Cannot execute a stub file.\n'
                'Original error: {0}'.format(e))
            raise InspectCudaException(msg)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def customize_compiler_for_nvcc(compiler, nvcc_settings):
    """inject deep into distutils to customize gcc/nvcc dispatch """

    # tell the compiler it can process .cu files
    compiler.src_extensions.append('.cu')

    # save references to the default compiler_so and _compile methods
    default_compiler_so = compiler.compiler_so
    default_compile = compiler._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # Use NVCC for .cu files
        if os.path.splitext(src)[1] == '.cu':
            compiler.set_executable('compiler_so', nvcc_settings['nvcc_path'])

        default_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        compiler.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    compiler._compile = _compile


def inspect_cuda():
    """ Return cuda device information and nvcc/cuda setup """
    nvcc_settings = nvcc_compiler_settings()
    sysconfig.get_config_vars()
    nvcc_compiler = ccompiler.new_compiler()
    sysconfig.customize_compiler(nvcc_compiler)
    customize_compiler_for_nvcc(nvcc_compiler, nvcc_settings)

    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)

    return json.loads(output), nvcc_settings
