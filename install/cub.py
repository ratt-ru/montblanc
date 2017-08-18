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
import shutil
import sys
import urllib2
import zipfile

from install_log import log

class InstallCubException(Exception):
    pass

def dl_cub(cub_url, cub_archive_name):
    """ Download cub archive from cub_url and store it in cub_archive_name """
    with open(cub_archive_name, 'wb') as f:
        remote_file = urllib2.urlopen(cub_url)
        meta = remote_file.info()

        # The server may provide us with the size of the file.
        cl_header = meta.getheaders("Content-Length")
        remote_file_size = int(cl_header[0]) if len(cl_header) > 0 else None

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
            log.warn("Local file size '{}' "
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