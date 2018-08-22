import numpy as np
import pytest
import tensorflow as tf

from montblanc.impl.rime.tensorflow.tensorflow_ops import (
                    sum_coherencies as sum_coherencies_op)


@pytest.mark.parametrize("FT, CT", [
    (np.float32, np.complex64),
    (np.float64, np.complex128),
])
@pytest.mark.parametrize("have_ant_1_scalar", [False, True])
@pytest.mark.parametrize("have_ant_2_scalar", [False, True])
@pytest.mark.parametrize("have_bl_scalar", [False, True])
@pytest.mark.parametrize("have_bl_jones", [False, True])
@pytest.mark.parametrize("have_base_coherencies", [False, True])
def test_sum_coherencies(FT, CT,
                         have_ant_1_scalar,
                         have_ant_2_scalar,
                         have_bl_scalar,
                         have_bl_jones,
                         have_base_coherencies,
                         tensorflow_gpu_devices):
    """ Implementation of the SumCoherencies operator test """

    def rf(*a, **kw):
        return np.random.random(*a, **kw).astype(FT)

    def rc(*a, **kw):
        return rf(*a, **kw) + 1j*rf(*a, **kw).astype(CT)

    from montblanc.impl.rime.tensorflow.rime_ops.op_test_utils import (
                                                        random_baselines)

    nsrc, ntime, na, nchan = 10, 15, 7, 16
    nbl = na*(na-1)//2

    chunks = np.random.random_integers(int(3.*nbl/4.), nbl, ntime)
    nvrow = np.sum(chunks)

    _, np_ant1, np_ant2, np_time_index = random_baselines(chunks, na)

    np_ant_scalar_1 = rc(size=(nsrc, ntime, na, nchan, 4))
    np_ant_jones_1 = rc(size=(nsrc, ntime, na, nchan, 4))
    np_ant_scalar_2 = rc(size=(nsrc, ntime, na, nchan, 4))
    np_ant_jones_2 = rc(size=(nsrc, ntime, na, nchan, 4))
    np_bl_scalar = rc(size=(nsrc, nvrow, nchan, 4))
    np_bl_jones = rc(size=(nsrc, nvrow, nchan, 4))
    np_base_coherencies = rc(size=(nvrow, nchan, 4))

    # Argument list
    np_args = [np_time_index, np_ant1, np_ant2,
               np_ant_scalar_1, np_ant_jones_1,
               np_bl_scalar, np_bl_jones,
               np_ant_scalar_2, np_ant_jones_2,
               np_base_coherencies]

    # Argument string name list
    arg_names = ['time_index', 'antenna1', 'antenna2',
                 'ant_scalar_1', 'ant_jones_1',
                 'baseline_scalar', 'baseline_jones',
                 'ant_scalar_2', 'ant_jones_2',
                 'base_coherencies']

    # These variables are optional and should be input as lists
    optionals = {'ant_scalar_1': have_ant_1_scalar,
                 'ant_scalar_2': have_ant_2_scalar,
                 'baseline_jones': have_bl_jones,
                 'baseline_scalar': have_bl_scalar,
                 'base_coherencies': have_base_coherencies}

    tf_args = [tf.Variable(v, name=n) if n not in optionals
               else [tf.Variable(v, name=n)] if optionals.get(n, False)
               else []
               for v, n in zip(np_args, arg_names)]

    # Compute expected result with numpy
    shape_2x2 = (nsrc, nvrow, nchan, 2, 2)

    if have_ant_1_scalar:
        ant_jones_1 = np_ant_scalar_1 * np_ant_jones_1
    else:
        ant_jones_1 = np_ant_jones_1

    if have_ant_2_scalar:
        ant_jones_2 = np_ant_scalar_2 * np_ant_jones_2
    else:
        ant_jones_2 = np_ant_jones_2

    if have_bl_jones and have_bl_scalar:
        bl_jones = np_bl_scalar * np_bl_jones
        mul_bl_jones = True
    elif have_bl_jones and not have_bl_scalar:
        bl_jones = np_bl_jones
        mul_bl_jones = True
    elif not have_bl_jones and have_bl_scalar:
        bl_jones = np_bl_scalar
        mul_bl_jones = True
    else:
        bl_jones = None
        mul_bl_jones = False

    ant1_jones = ant_jones_1[:, np_time_index, np_ant1]
    ant2_jones = ant_jones_2[:, np_time_index, np_ant2].conj()
    tshape = (0, 1, 2, 4, 3)

    if mul_bl_jones:
        expected = np.einsum("srcij,srcjk,srckl->rcil",
                             ant1_jones.reshape(shape_2x2),
                             bl_jones.reshape(shape_2x2),
                             ant2_jones.reshape(shape_2x2).transpose(tshape))
    else:
        expected = np.einsum("srcij,srcjk->rcik",
                             ant1_jones.reshape(shape_2x2),
                             ant2_jones.reshape(shape_2x2).transpose(tshape))

    expected = expected.reshape(nvrow, nchan, 4)

    # Add base coherencies
    if have_base_coherencies:
        expected += np_base_coherencies

    def _pin_op(device, *tf_args):
        """ Pin operation to device """
        with tf.device(device):
            return sum_coherencies_op(*tf_args, FT=FT)

    # Pin operation to CPU
    cpu_op = _pin_op('/cpu:0', *tf_args)

    # Run the op on all GPUs
    gpu_ops = [_pin_op(d, *tf_args) for d in tensorflow_gpu_devices]

    # Initialise variables
    init_op = tf.global_variables_initializer()

    with tf.Session() as S:
        S.run(init_op)

        # Get the CPU coherencies
        cpu_coh = S.run(cpu_op)
        assert np.allclose(expected, cpu_coh)

        # Parametrize this if necessary
        cmp_kw = {}

        # Compare against the GPU coherencies
        for gpu_coh in S.run(gpu_ops):
            if not np.allclose(cpu_coh, gpu_coh, **cmp_kw):
                if FT == np.float32:
                    pytest.fail("CPU and GPU results don't match for "
                                "single precision float data. Consider "
                                "relaxing the tolerance")
                else:
                    pytest.fail("CPU and GPU results don't match!")
