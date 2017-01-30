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

import textwrap

_desc_wrapper = textwrap.TextWrapper(initial_indent=" "*4,
    subsequent_indent=" "*8, width=70)

_help_wrapper = textwrap.TextWrapper(width=70)

def context_help(context, display_cube=False):
    from montblanc.impl.rime.tensorflow.sources.source_context import SourceContext
    from montblanc.impl.rime.tensorflow.sinks.sink_context import SinkContext

    if isinstance(context, SourceContext):
        ctx_type = 'source'
        behaviour1 = 'requesting'
        behaviour2 = 'return'
        shape = context.shape
        dtype = context.dtype
    elif isinstance(context, SinkContext):
        ctx_type = 'sink'
        behaviour1 = 'providing'
        behaviour2 = 'with'
        shape = context.data.shape
        dtype = context.data.dtype
    else:
        raise TypeError("Invalid context {t}".format(t=type(context)))

    description = context._array_schema.get('description', 'No Description')
    units = context._array_schema.get('units', 'None Specified')
    schema = context._array_schema.shape
    cube = context._cube
    cube_dims = cube.dimensions(copy=False)
    global_shape = tuple([cube.dim_global_size(d) if d in cube_dims
        else d for d in schema])
    l_extents = tuple([cube.dim_lower_extent(d) if d in cube_dims
        else 0 for d in schema])
    u_extents = tuple([cube.dim_upper_extent(d) if d in cube_dims
        else d for d in schema])

    dim_pad = " "*12
    wrap = _desc_wrapper.wrap

    lines = []
    lines.append("'{name}' data source information:".format(
        name=context._name))
    lines += wrap("Description: {description}".format(
        description=description))
    lines += wrap("Units: {units}".format(units=units))
    lines += wrap("Schema or abstract shape: {schema}\n".format(schema=schema))
    lines += ["{p}where '{s}' is '{d}'".format(
            p=dim_pad, s=d, d=cube_dims[d].description)
        for d in schema if d in cube_dims]
    lines += wrap("Global shape on this iteration: "
        "{global_shape}\n".format(global_shape=global_shape))
    lines += wrap("Local shape for this context: "
        "{local_shape}\n".format(local_shape=shape))
    lines += wrap("Lower extents within global shape: "
        "{lower_extents}\n".format(lower_extents=l_extents))
    lines += wrap("Upper extents within global shape: "
        "{upper_extents}\n".format(upper_extents=u_extents))
    lines.append('\n')

    dims, strides = zip(*context._iter_args)

    lines.append("Iteration information:")
    lines += wrap("Iterating over the {d} "
        "dimensions with global sizes of {gs} "
        "in strides of {s}.".format(
            d=dims, s=strides, gs=tuple(cube.dim_global_size(*dims))))
    lines.append('\n')

    wrap = _help_wrapper.wrap

    lines += wrap("This context is {b1} the '{name}' data {ctype} "
        "{b2} an array of shape '{shape}' and dtype '{dtype}'. "
        "This portion of data lies between the lower '{lex}' "
        " and upper '{uex}' extent of a global shape of '{gs}'. "
        "The abstract shape of this data source is {schema}.".format(
            name=context._name, ctype=ctx_type,
            b1=behaviour1, b2=behaviour2,
            shape=shape, dtype=dtype,
            schema=schema, lex=l_extents, uex=u_extents, gs=global_shape))

    if display_cube is True:
        lines += ['', 'Hypercube', '', str(cube)]

    return '\n'.join(lines)
