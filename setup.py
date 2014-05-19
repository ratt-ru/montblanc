# This setup script is based on Robert McGibbon's stackoverflow
# answer here: http://stackoverflow.com/a/13300714

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

predict_ext = Extension('predict',
	sources=['montblanc/predict.c'],
	include_dirs=[numpy_include],
	extra_compile_args={
		'gcc': ['-fopenmp']}
)

crimes_ext = Extension('crimes',
	sources=[
		'montblanc/moderngpu/src/mgpuutil.cpp',
		'montblanc/moderngpu/src/mgpucontext.cu',
		'montblanc/crimes.cu',
	],
	library_dirs=[CUDA['lib64']],
	libraries=['cudart'],
	runtime_library_dirs=[CUDA['lib64']],
	# this syntax is specific to this build system
	# we're only going to use certain compiler args
	# with nvcc and not with gcc
	# the implementation of this trick is in
	# customize_compiler() below
	extra_compile_args={
		'gcc': [],
	    'nvcc':
	    	[
	    	 '-c',
	    	 '--compiler-options',
	    	 "'-fPIC'",                                    	
	    	 '--ptxas-options=-v',
	    	 '-gencode',
	    	 'arch=compute_20,code=sm_20',
	    	 '-gencode',
	    	 'arch=compute_30,code=sm_30',
	    	 '-gencode',
	    	 'arch=compute_35,code=sm_35',
	    	 ]},
	include_dirs = [
		numpy_include,
		'montblanc/moderngpu/include',
		CUDA['include']]
)

def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

def readme():
	with open('README.md') as f:
		return f.read()

setup(name='montblanc',
      version='0.1',
      description='GPU-accelerated RIME implementations.',
      long_description=readme(),
      url='http://github.com/ska-sa/montblanc',
      author='Simon Perkins',
      author_email='simon.perkins@gmail.com',
      license='MIT',
      packages=[
      	'montblanc',
      	'montblanc.examples'],
      install_requires=[
      	'numpy',
      	'pycuda',
      	'pyrap',
      ],

      ext_modules = [crimes_ext, predict_ext],

      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},      
      zip_safe=False)