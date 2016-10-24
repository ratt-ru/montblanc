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

import hashlib
import os
import urllib2
import shutil
import subprocess
import sys
import zipfile
from setuptools import setup, find_packages

mb_path = 'montblanc'
mb_inc_path = os.path.join(mb_path, 'include')

def dl_cub(cub_url, cub_archive_name):
    """ Download cub archive from cub_url and store it in cub_archive_name """
    with open(cub_archive_name, 'wb') as f:
        remote_file = urllib2.urlopen(cub_url)
        meta = remote_file.info()

        # The server may provide us with the size of the file.
        cl_header = meta.getheaders("Content-Length")
        remote_file_size = int(cl_header[0]) if len(cl_header) > 0 else '???'

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

            status = r"Downloading %s %10d/%s" % (cub_url,
                local_file_size, remote_file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

        remote_file.close()

def sha_hash_file(filename):
    # Compute the SHA1 hash
    hash_sha = hashlib.sha1()

    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            hash_sha.update(chunk)

    return hash_sha.hexdigest()

def is_cub_installed(readme_filename, header_filename, cub_version_str):
    # Check if the cub.h exists
    if not os.path.exists(header_filename) or not os.path.isfile(header_filename):
        return False

    # Check if the README.md exists
    if not os.path.exists(readme_filename) or not os.path.isfile(readme_filename):
        return False

    # Search for the version string, returning True if found
    with open(readme_filename, 'r') as f:
        for line in f:
            if line.find(cub_version_str) != -1:
                return True

    # Nothing found!
    return False

def install_cub():
    """ Downloads and installs cub """
    cub_url = 'https://github.com/NVlabs/cub/archive/1.5.2.zip'
    cub_sha_hash = 'b98dabe346c5e1ab24db250379d73afe14189055'
    cub_version_str = 'Current release: v1.5.2 (03/21/2016)'
    cub_zip_file = 'cub.zip'
    cub_zip_dir = 'cub-1.5.2'
    cub_unzipped_path = os.path.join(mb_inc_path, cub_zip_dir)
    cub_new_unzipped_path = os.path.join(mb_inc_path, 'cub')
    cub_header = os.path.join(cub_new_unzipped_path, 'cub', 'cub.cuh')
    cub_readme = os.path.join(cub_new_unzipped_path, 'README.md' )

    # Check for a reasonably valid install
    cub_installed = is_cub_installed(cub_readme, cub_header, cub_version_str)
    print 'NVIDIA cub installed: %s' % cub_installed
    if cub_installed:
        return

    # Do we already have a valid cub zip file
    have_valid_cub_file = (os.path.exists(cub_zip_file) and
        os.path.isfile(cub_zip_file) and
        sha_hash_file(cub_zip_file) == cub_sha_hash)

    print 'NVIDIA cub archive downloaded: %s' % have_valid_cub_file

    # Download if we don't have a valid file
    if not have_valid_cub_file:
        dl_cub(cub_url, cub_zip_file)
        cub_file_sha_hash = sha_hash_file(cub_zip_file)

        # Compare against our supplied hash
        assert cub_sha_hash == cub_file_sha_hash, \
             ('Hash of file %s downloaded from %s '
            'is %s and does not match the expected '
            'hash of %s. Please manually download '
            'as per the README.md instructions.') % (cub_zip_file,
                cub_url, cub_sha_hash, cub_file_sha_hash)

    # Unzip into montblanc/include/cub
    with zipfile.ZipFile(cub_zip_file, 'r') as zip_file:
        # Remove any existing installs
        shutil.rmtree(cub_unzipped_path, ignore_errors=True)
        shutil.rmtree(cub_new_unzipped_path, ignore_errors=True)

        # Unzip
        zip_file.extractall(mb_inc_path)

        # Rename
        shutil.move(cub_unzipped_path, cub_new_unzipped_path)

        print 'NVIDIA cub archive unzipped into %s' % cub_new_unzipped_path

    assert is_cub_installed(cub_readme, cub_header, cub_version_str),  \
        ('cub installed unexpectedly failed!')

    print 'NVIDIA cub install successful'

install_cub()

def get_version():
    # Versioning code here, based on
    # http://blogs.nopcode.org/brainstorm/2013/05/20/pragmatic-python-versioning-via-setuptools-and-git-tags/

    # Fetch version from git tags, and write to version.py.
    # Also, when git is not available (PyPi package), use stored version.py.
    version_py = os.path.join('montblanc', 'version.py')

    try:
        version_git = subprocess.check_output(['git', 'describe', '--tags']).rstrip()
    except:
        with open(version_py, 'r') as fh:
            version_git = open(version_py).read().strip().split('=')[-1].replace('"','')

    version_msg = "# Do not edit this file, pipeline versioning is governed by git tags"

    with open(version_py, 'w') as fh:
        fh.write(version_msg + os.linesep + "__version__=\"" + version_git +"\"")

    return version_git

def readme():
    with open('README.rst') as f:
        return f.read()

def include_pkg_dirs():
    """
    Recursively provide package_data directories for
    directories in montblanc/include.
    """
    pkg_dirs = []

    l = len(mb_path) + len(os.sep)
    # Ignore
    exclude = ['docs', '.git', '.svn']

    # Walk 'montblanc/include'
    for root, dirs, files in os.walk(mb_inc_path, topdown=True):
        # Prune out everything we're not interested in
        # from os.walk's next yield.
        dirs[:] = [d for d in dirs if d not in exclude]

        for d in dirs:
            # OK, so everything starts with 'montblanc/'
            # Take everything after that ('include...') and
            # append a '/*.*' to it
            pkg_dirs.append(os.path.join(root[l:], d, '*.*'))

    return pkg_dirs

setup(name='montblanc',
    version=get_version(),
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
    license='GPL2',
    packages=find_packages(),
    install_requires=[
        'astropy >= 1.2.1',
        'attrdict >= 2.0.0',
        'enum34 >= 1.1.2',
        'funcsigs >= 0.4',
        'futures >= 3.0.3',
        'hypercube >= 0.3.0a3',
        'numpy >= 1.9.2',
        'numexpr >= 2.4',
        'python-casacore >= 2.1.2',
    ],
    setup_requires=['numpy >= 1.9.2'],
    package_data={'montblanc': include_pkg_dirs()},
    include_package_data=True,
    zip_safe=False)
