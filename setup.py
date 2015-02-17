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

import os
from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


def src_pkg_dirs():
    """
    Recursively provide package_data directories for
    directories in montblanc/src.
    """
    pkg_dirs = []

    mbdir = 'montblanc'
    l = len(mbdir) + len(os.sep)
    path = os.path.join(mbdir, 'src')
    # Ignore
    exclude = ['docs', '.git', '.svn']

    # Walk 'montblanc/src'
    for root, dirs, files in os.walk(path, topdown=True):
        # Prune out everything we're not interested in
        # from os.walk's next yield.
        dirs[:] = [d for d in dirs if d not in exclude]

        for d in dirs:
            # OK, so everything starts with 'montblanc/'
            # Take everything after that ('src...') and
            # append a '/*.*' to it
            pkg_dirs.append(os.path.join(root[l:], d, '*.*'))

    return pkg_dirs

setup(name='montblanc',
    version='0.1',
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
    license='MIT',
    packages=[
        'montblanc',
        'montblanc.api',
        'montblanc.api.loaders',
        'montblanc.examples',
        'montblanc.ext',
        'montblanc.impl',
        'montblanc.impl.biro',
        'montblanc.impl.biro.v2',
        'montblanc.impl.biro.v2.gpu',
        'montblanc.impl.biro.v2.cpu',
        'montblanc.impl.biro.v2.loaders',
        'montblanc.impl.biro.v3',
        'montblanc.impl.biro.v3.gpu',
        'montblanc.impl.biro.v4',
        'montblanc.impl.biro.v4.gpu',
        'montblanc.impl.biro.v4.cpu',
        'montblanc.impl.biro.v4.loaders',
        'montblanc.impl.biro.v5',
        'montblanc.impl.biro.v5.gpu',
        'montblanc.impl.common',
        'montblanc.impl.common.loaders',
        'montblanc.tests',
        'montblanc.util'],
    install_requires=[
        'numpy',
        'numexpr',
        'pycuda',
        'pytools',
    ],
    package_data={
        'montblanc': ['log/*.json'],
        'montblanc': src_pkg_dirs()},
    include_package_data=True,
    zip_safe=False)
