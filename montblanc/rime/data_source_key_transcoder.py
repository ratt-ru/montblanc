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

import math

import attr
import bitstring as bs

_T = attr.make_class("DataSourceEncodingData", ["data_source",
                                                "partition",
                                                "id_in_part",
                                                "count",
                                                "vdims"])

class DataSourceKeyTranscoder(object):
    """
    Encodes a data source and the tile ids of its varying dimensions
    into a unique 64 bit key, suitable for use as a key
    in a tensorflow StagingArea.
    """

    def __init__(self, data_source_partition, varying_dims):
        """
        Constructs a DataSourceKeyTranscoder

        Parameters
        ----------
            data_source_partition : dict
                A { partition : [data_sources] } mapping
            varying_dims : iterable of str
                A sequence of varying dimension names
        """
        dsp = attr.asdict(data_source_partition)
        self._varying_dims = varying_dims = set(varying_dims)

        def _ds_vdim(ds):
            """ Return (correctly ordered) varying dims for this data source """
            return [s for s in ds.shape if s in varying_dims]

        # Data source name to partition map
        self._ds_map = { ds.name: _T(ds, p, i, len(ds_list), _ds_vdim(ds))
                                                for p, ds_list in dsp.items()
                                                for i, ds in enumerate(ds_list) }

        self._ds_pack_fmt = None

    @property
    def vdim_max_tiles(self):
        raise NotImplementedError("Gets aren't supported for this property")

    @vdim_max_tiles.setter
    def vdim_max_tiles(self, max_tiles):
        """ Sets the maximum number of tiles for each varying dimension """

        if not set(max_tiles.keys()) == self._varying_dims:
            raise ValueError("Not all dimensions '{}' were specified '{}'"
                .format(max_tiles.keys(), list(self._varying_dims)))

        def _format(ds):
            """
            Create a format string for the given data source that looks like the following
            "data source id, vdim0, vdim1, ..., vdimn, padding"
            """

            # Encode a constant binary string for the data source id
            fmt_parts = ["0b" + bs.pack("uint:%s=%s" % (ds.count, ds.id_in_part)).bin]
            nbits = ds.count

            # For each variable dimension, encode a time tile (8) and token (ntime)
            # e.g. "uint:8=ntime"
            for d in ds.vdims:
                try:
                    dim_max_tiles = max_tiles[d]
                except KeyError as e:
                    raise ValueError("No dimension size supplied "
                                     "for '{}'".format(d))

                bits = int(math.ceil(math.log(dim_max_tiles, 2)))
                fmt_parts.append("uint:%s=%s" % (bits, d))
                nbits += bits

            remaining_bits = 64 - nbits

            if remaining_bits < 0:
                raise ValueError("Couldn't pack data source '{}' "
                                "and its varying dimensions '{}' "
                                "into 64 bits!".format(ds.name, ds.vdims))

            # Zero pad remaining bits
            fmt_parts.append("0b" + bs.pack("uint:%s=0" % remaining_bits).bin)

            return ", ".join(fmt_parts)

        
        self._ds_pack_fmt = { t.data_source.name : _format(t)
                               for t in self._ds_map.values() }

    def encode(self, data_source, **vdims):
        """
        Return a unique 64 int for the given data_source
        and varying dimension tile id's within the data source's
        partition.

        Parameters
        ----------
            data_source : str
                Name of the data source
            **vdims
                Keywords in the form of dim=tile_id

        Returns
        -------
        int
            unique 64 bit integer describing the data source
            and varying tile id's within the data source's partition.
        """
        try:
            ds_tup = self._ds_map[data_source]
        except AttributeError as e:
            raise AttributeError("Data source '{}' not configured "
                                "in this transcoder!".format(data_source))

        try:
            pack_fmt = self._ds_pack_fmt[data_source]
        except TypeError as e:
            if self._ds_pack_fmt is None:
                raise ValueError("Set varying dimension sizes "
                                 "with vdim_max_tiles.")

            raise e
        except AttributeError as e:
            raise AttributeError("Bit configuration for data source "
                                 "'{}' not present".format(data_source))

        if not len(vdims) == len(ds_tup.vdims):
            raise ValueError("Invalid dimension data '{}'. "
                            "Data should be provided for the "
                            "following dimensions '{}'".format(dims, ds_tup.vdims))

        return bs.pack(pack_fmt, **vdims).int
