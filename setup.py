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
import inspect
import itertools
import json
import glob
import hashlib
import logging
import os
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
import shutil
import subprocess
import sys

try:
    import urllib.request, urllib.error, urllib.parse
except ImportError:
    import urllib2 as urllib

import tempfile
import zipfile


PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

# =============
# Setup logging
# =============

log_format = "%(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger("Montblanc Install")

global device_info, nvcc_settings
device_info = None
nvcc_settings = None

minimum_cuda_version = 8000

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
        log.warning("nvcc compiler not found at '{}'. "
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

        log.warning("CUDA_PATH not found, inferring it as '{}' "
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
        raise InspectCudaException(msg)

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

def customize_compiler_for_nvcc_inspection(compiler, nvcc_settings):
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
    customize_compiler_for_nvcc_inspection(nvcc_compiler, nvcc_settings)

    output = inspect_cuda_version_and_devices(nvcc_compiler, nvcc_settings)

    return json.loads(output), nvcc_settings

class InstallCubException(Exception):
    pass

def dl_cub(cub_url, cub_archive_name):
    """ Download cub archive from cub_url and store it in cub_archive_name """
    with open(cub_archive_name, 'wb') as f:
        remote_file = urllib.request.urlopen(cub_url)
        meta = remote_file.info()

        # The server may provide us with the size of the file.
        cl_header = meta.get("Content-Length")
        remote_file_size = int(cl_header[0]) if cl_header is not None and len(cl_header) > 0 else None

        # Initialise variables
        local_file_size = 0
        block_size = 128*1024

        # Do the download
        while True:
            data = remote_file.read(block_size)

            if not data:
                break

            f.write(data)
            local_file_size += len(data)

        if (remote_file_size is not None and
                not local_file_size == remote_file_size):
            log.warning("Local file size '{}' "
                "does not match remote '{}'".format(
                    local_file_size, remote_file_size))

        remote_file.close()

def sha_hash_file(filename):
    """ Compute the SHA1 hash of filename """
    hash_sha = hashlib.sha1()

    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            hash_sha.update(chunk)

    return hash_sha.hexdigest()

def is_cub_installed(readme_filename, header_filename, cub_version_str):
    # Check if the cub.h exists
    if not os.path.exists(header_filename) or not os.path.isfile(header_filename):
        reason = "CUB header '{}' does not exist".format(header_filename)
        return (False, reason)

    # Check if the README.md exists
    if not os.path.exists(readme_filename) or not os.path.isfile(readme_filename):
        reason = "CUB readme '{}' does not exist".format(readme_filename)
        return (False, reason)

    # Search for the version string, returning True if found
    with open(readme_filename, 'r') as f:
        for line in f:
            if line.find(cub_version_str) != -1:
                return (True, "")

    # Nothing found!
    reason = "CUB version string '{}' not found in '{}'".format(
        cub_version_str, readme_filename)
    return (False, reason)

def install_cub(mb_inc_path):
    """ Downloads and installs cub into mb_inc_path """
    cub_url = 'https://github.com/NVlabs/cub/archive/1.6.4.zip'
    cub_sha_hash = '0d5659200132c2576be0b3959383fa756de6105d'
    cub_version_str = 'Current release: v1.6.4 (12/06/2016)'
    cub_zip_file = 'cub.zip'
    cub_zip_dir = 'cub-1.6.4'
    cub_unzipped_path = os.path.join(mb_inc_path, cub_zip_dir)
    cub_new_unzipped_path = os.path.join(mb_inc_path, 'cub')
    cub_header = os.path.join(cub_new_unzipped_path, 'cub', 'cub.cuh')
    cub_readme = os.path.join(cub_new_unzipped_path, 'README.md' )

    # Check for a reasonably valid install
    cub_installed, _ = is_cub_installed(cub_readme, cub_header, cub_version_str)
    if cub_installed:
        log.info("NVIDIA cub installation found "
            "at '{}'".format(cub_new_unzipped_path))
        return

    log.info("No NVIDIA cub installation found")

    # Do we already have a valid cub zip file
    have_valid_cub_file = (os.path.exists(cub_zip_file) and
        os.path.isfile(cub_zip_file) and
        sha_hash_file(cub_zip_file) == cub_sha_hash)

    if have_valid_cub_file:
        log.info("Valid NVIDIA cub archive found '{}'".format(cub_zip_file))
    # Download if we don't have a valid file
    else:
        log.info("Downloading cub archive '{}'".format(cub_url))
        dl_cub(cub_url, cub_zip_file)
        cub_file_sha_hash = sha_hash_file(cub_zip_file)

        # Compare against our supplied hash
        if cub_sha_hash != cub_file_sha_hash:
            msg = ('Hash of file %s downloaded from %s '
                'is %s and does not match the expected '
                'hash of %s. Please manually download '
                'as per the README.md instructions.') % (
                    cub_zip_file, cub_url,
                    cub_file_sha_hash, cub_sha_hash)

            raise InstallCubException(msg)

    # Unzip into montblanc/include/cub
    with zipfile.ZipFile(cub_zip_file, 'r') as zip_file:
        # Remove any existing installs
        shutil.rmtree(cub_unzipped_path, ignore_errors=True)
        shutil.rmtree(cub_new_unzipped_path, ignore_errors=True)

        # Unzip
        zip_file.extractall(mb_inc_path)

        # Rename. cub_unzipped_path is mb_inc_path/cub_zip_dir
        shutil.move(cub_unzipped_path, cub_new_unzipped_path)

        log.info("NVIDIA cub archive unzipped into '{}'".format(
            cub_new_unzipped_path))


    there, reason = is_cub_installed(cub_readme, cub_header, cub_version_str)

    if not there:
        raise InstallCubException(reason)


tensorflow_extension_name = 'montblanc.ext.rime'

def customize_compiler_for_tensorflow(compiler, nvcc_settings, device_info, 
                                      march_native=False, gcc_verbosity="",
                                      linker_options=""):
    """inject deep into distutils to customize gcc/nvcc dispatch """
    compiler_verbosity_flags = list(map(lambda x: x.strip(), gcc_verbosity.split(" ")))
    if compiler_verbosity_flags == [""]:
        compiler_verbosity_flags = []
    linker_options = list(map(lambda x: x.strip(), linker_options.split(" ")))
    if linker_options == [""]:
       linker_options = None
    if march_native:
        log.warn("Warning: native marching enabled - the binaries are NOT PORTABLE\n"
                 "Disable this option before building distributed images/wheels")
        opt_flags = ['-march=native', '-mtune=native']
        opt_flags += compiler_verbosity_flags
    else:
        opt_flags = [] + compiler_verbosity_flags
    # tell the compiler it can process .cu files
    compiler.src_extensions.append('.cu')
    # save references to the default compiler_so and _comple methods
    default_compiler_so = compiler.compiler_so
    default_compile = compiler._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    if linker_options and len(linker_options) > 0:
        log.warn("Warning: overrriding default linker options \"[{}]\" with \"[{}]\"".format(
                 " ".join(compiler.linker_so), " ".join(linker_options)))
        compiler.linker_so = linker_options
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu' and nvcc_settings is not None:
            # use the cuda for .cu files
            compiler.set_executable('compiler_so', nvcc_settings['nvcc_path'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc'] + [val for pair in zip(["--compiler-options"]*len(opt_flags),
                                                                      map(lambda x: '"{}"'.format(x), opt_flags))
                                                 for val in pair]
        else:
            postargs = extra_postargs['gcc'] + opt_flags

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
    import tensorflow.compat.v1 as tf
    import glob

    use_cuda = nvcc_settings is not None and (bool(nvcc_settings['cuda_available'])
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
    libraries = []
    for ldir in library_dirs:
        libraries += map(lambda x: ":{}".format(x), 
                         map(os.path.basename, 
                             glob.glob(os.path.join(ldir, "*.so*"))))
    #libraries = [':libtensorflow_framework.so.2']
    extra_link_args = ['-fPIC', '-fopenmp']

    # Macros
    define_macros = [
        ('_MWAITXINTRIN_H_INCLUDED', None),
        ('_FORCE_INLINES', None),
        ('_GLIBCXX_USE_CXX11_ABI', 0)]

    # Common flags
    flags = ['-std=c++14']

    gcc_flags = flags + ['-fPIC', '-fopenmp']
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

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext

class BuildCommand(build_ext):
    """ Custom build command for building the tensorflow extension """
    user_options = build_ext.user_options + [
        ('march-native=', None, 'Enable native marching (optimized, non-portable binaries)'),
        ('compiler-verbosity=', None, 'Control GCC compiler verbosity'),
        ('linker-options=', None, 'Overrulling options to linker - this overrides all defaults')
    ]
    def get_ext_filename(self, ext_name):
        if PY3:
            filename = super().get_ext_filename(ext_name)
            return get_ext_filename_without_platform_suffix(filename)
        else:
            return build_ext.get_ext_filename(self, ext_name)

    def initialize_options(self):
        build_ext.initialize_options(self)
        global device_info, nvcc_settings
        self.nvcc_settings = nvcc_settings
        self.cuda_devices = device_info
        self.march_native = False
        self.compiler_verbosity = None
        self.linker_options = None

    def build_extensions(self):
        if isinstance(self.march_native, str):
            if self.march_native.upper() == "TRUE" or \
            self.march_native == "ON" or \
            self.march_native == "YES":
                march_native = True
            elif self.march_native.upper() == "FALSE" or \
                self.march_native == "OFF" or \
                self.march_native == "NO":
                march_native = False
            else:
                raise ValueError("Option march_native must be type boolean (true / on / yes) or converse acceptable")
        elif isinstance(self.march_native, bool):
            march_native = self.march_native
        else:
            raise ValueError("Option march_native is neither string or boolean")
        if self.compiler_verbosity is None:
            self.compiler_verbosity = ""
        if self.linker_options is None:
            self.linker_options = ""

        customize_compiler_for_tensorflow(self.compiler,
            self.nvcc_settings, self.cuda_devices, 
            march_native=march_native,
            gcc_verbosity=self.compiler_verbosity,
            linker_options=self.linker_options)
        build_ext.build_extensions(self)

# ==================
# setuptools imports
# ==================

from distutils.version import LooseVersion
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.dist import Distribution

mb_path = 'montblanc'
mb_inc_path = os.path.join(mb_path, 'include')

# =================
# Detect readthedocs
# ==================

on_rtd = os.environ.get('READTHEDOCS') == 'True'

# Inspect previous tensorflow installs
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    log.error("No version of tensorflow discovered. "
              "You should install this package using pip, "
              "not python setup.py install or develop (standard PEP 518 - "
              "you should not be seeing this message)")

    tf_installed = False
    use_tf_cuda = False
else:
    # setuptools will handle version clashes
    tf_installed = True
    use_tf_cuda = tf.test.is_built_with_cuda()

# ===========================
# Detect CUDA and GPU Devices
# ===========================

# See if CUDA is installed and if any NVIDIA devices are available
# Choose the tensorflow flavour to install (CPU or GPU)
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
    'hypercube == 0.3.4',
    'tensorflow >= 2.7.0,<2.8; python_version >="3.8"', #ubuntu 20.04 with distro nvcc/gcc
    'tensorflow <=2.4.4; python_version <"3.8"', #ubuntu 18.04 with distro nvcc/gcc
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

    if not tf_installed: #error raised earlier
        sys.exit(1)

    cmdclass = {'build_ext': BuildCommand}
    # tensorflow_ops_ext.BuildCommand.run will
    # expand this dummy extension to its full portential
    ext_modules = [create_tensorflow_extension(nvcc_settings, device_info)]

    # Pass NVCC and CUDA settings through to the build extension
    ext_options = {
        'build_ext': {
            'nvcc_settings': nvcc_settings,
            'cuda_devices': device_info
        },
    }

log.info('install_requires={}'.format(install_requires))

setup(name='montblanc',
    version="0.7.2.1",
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
    python_requires='>=3.6',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    options=ext_options,
    license='GPL2',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False)
