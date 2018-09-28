from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from itertools import product

import numpy as np
import tensorflow as tf
import pytest

from montblanc.impl.rime.tensorflow.tensorflow_ops import (
                    jones_multiply as jones_multiply_op)

Analysis = namedtuple("Analysis", ["tf_shape", "tf_schema",
                                   "ein_shape", "ein_schema"])


@pytest.mark.parametrize("FT, CT", [
    (np.float32, np.complex64),
    (np.float64, np.complex128),
])
@pytest.mark.parametrize("in_shape, out_shape, squeeze", [
    [("stafij", "tfjk", "sakl"), ("stafil",), False],
    [("stafij", "tafjk", "sakl"), ("stafil",), True],
    [("afij", "tfjk", "sakl"), ("stafil",), False],
    [("ij", "tfjk", "sakl"), ("stafil",), True],
    [("ij", "tf", "sajl"), ("stafil",), True],
    [("ij", "tfjk", "sakl", "staflm"), ("stafim",), False],
    [("ij", "tfjk"), ("tfik",), True],
    [("aij", "tjk"), ("taik",), True],
    [("rij", "srfjk"), ("srfik",), True]
])
def test_jones_multiply(FT, CT, in_shape, out_shape, squeeze,
                        tensorflow_gpu_devices):
    """ Implementation of the JonesMultiply operator test """

    def rf(*args, **kwargs):
        return np.random.random(*args, **kwargs).astype(FT)

    def rc(*args, **kwargs):
        return rf(*args, **kwargs).astype(CT)

    corr_dims = ['i', 'j', 'k', 'l', 'm']
    corr_prods = (((c1+c2), (c2+c1))
                  for c1, c2
                  in product(corr_dims, corr_dims)
                  if c1 != c2)

    # Produces a unique set of pair correlation indices
    # { 'ij', 'ik', ..., 'li'}
    corrs = set(c for sublist in corr_prods for c in sublist)

    dim_sizes = {
        's': 5,
        't': 10,
        'r': 4,
        'a': 7,
        'f': 16,
    }

    # All correlations will have dimension 4
    dim_sizes.update({c: 4 for c in corrs})

    einsum_dim_to_schema = [
        ('s', 'source'),
        ('r', 'row'),
        ('t', 'time'),
        ('a', 'ant'),
        ('f', 'chan'),
    ]

    # Map all correlation pairs to the 'corr' dimension
    einsum_dim_to_schema.extend([(c, 'corr') for c in corrs])

    def _analyse(einsum_schemas):
        for einsum_schema in einsum_schemas:
            schema = []
            einsum_shape = []
            tf_shape = []

            for e, dim in einsum_dim_to_schema:
                i = einsum_schema.find(e)

                if i != -1:
                    schema.append(dim)

                    if len(e) == 1:
                        einsum_shape.append(dim_sizes[e])
                        tf_shape.append(dim_sizes[e])
                    elif len(e) == 2:
                        # Handle correlations
                        ds = dim_sizes[e]
                        assert ds == 4
                        einsum_shape.append(2)
                        einsum_shape.append(2)
                        tf_shape.append(ds)
                    else:
                        raise ValueError("dims must be length 1 or 2")

            schema = "".join(("(", ",".join(schema), ")"))

            yield Analysis(tuple(tf_shape), schema,
                           tuple(einsum_shape), einsum_schema)

    input_analysis = list(_analyse(in_shape))
    output_analysis = list(_analyse(out_shape))

    # Create input variables
    # Argument list
    np_args = [rc(size=(a.tf_shape)) for a in input_analysis]
    schemas = [a.tf_schema for a in input_analysis]

    # Argument string name list
    # Constructor tensorflow variables
    tf_args = [[tf.Variable(v) for v in np_args]]
    tf_kwargs = {'schemas': schemas, 'FT': FT, 'squeeze': squeeze,
                 'output_schema': output_analysis[0].tf_schema}

    def _pin_op(device, *tf_args, **tf_kwargs):
        """ Pin operation to device """
        with tf.device(device):
            return jones_multiply_op(*tf_args, **tf_kwargs)

    # Pin operation to CPU
    cpu_op = _pin_op('/cpu:0', *tf_args, **tf_kwargs)

    # Run the op on all GPUs
    gpu_ops = [_pin_op(d, *tf_args, **tf_kwargs)
               for d in tensorflow_gpu_devices]

    # Initialise variables
    init_op = tf.global_variables_initializer()

    with tf.Session() as S:
        S.run(init_op)
        cpu_result = S.run(cpu_op)

        # Construct einsum expression
        einsum_expr = ",".join([a.ein_schema for a in input_analysis])
        einsum_expr = "->".join((einsum_expr, output_analysis[0].ein_schema))

        # Construct einsum inputs
        einsum_inputs = [var.reshape(a.ein_shape) for var, a
                         in zip(np_args, input_analysis)]

        # Compute einsum
        np_result = np.einsum(einsum_expr, *einsum_inputs)
        np_result = np_result.reshape(output_analysis[0].tf_shape)

        # Check CPU result
        assert np.allclose(np_result, cpu_result)

        for gpu_result in S.run(gpu_ops):
            assert np.allclose(cpu_result, gpu_result)
