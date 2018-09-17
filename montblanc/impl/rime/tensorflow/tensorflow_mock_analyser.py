from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import contextlib
from functools import partial
import inspect
import logging

import tensorflow as tf

from montblanc.impl.rime.tensorflow.tensorflow_ops import (op_defs,
                                                           parse_shape_schema)
from montblanc.impl.rime.tensorflow.map_dataset import (TensorMap,
                                                        MapDataset)
from montblanc.impl.rime.tensorflow.queue_dataset import (TensorQueue,
                                                          QueueDataset)


mock = tf.test.mock

log = logging.getLogger("__name__")


class KnownVariable(object):
    """ Indicates a variable which we know about """
    pass


class UnknownVariable(object):
    """ Indicates a variable of which we know nothing """
    pass


class PlaceholderVariable(object):
    """ Indicates a placeholder variable """
    pass


def arg_type_info(arg_def, op_def):
    """ Figure out argument type information """
    if arg_def.type:
        # Fixed type, easy
        dtype = tf.as_dtype(arg_def.type)
        type_name = dtype.name
        allowed = [dtype]
    elif arg_def.type_attr:
        # If a polymorphic type, there'll be an attribute
        # with a default type associated
        type_name = arg_def.type_attr
        type_attr = op_def.attr[arg_def.type_attr]
        allowed = type_attr.allowed_values.list
        allowed = [tf.as_dtype(dt) for dt in allowed.type]
        dtype = tf.as_dtype(type_attr.default_value.type)
    elif arg_def.type_list_attr:
        # Implement me
        raise ValueError("Type Lists not handled")
    else:
        raise TypeError("Couldn't infer type "
                        "of argument %s" % arg_def.name)

    return {
        'allowed_types': allowed,
        'default': dtype,
        'default_type_name': type_name,
    }


def arg_schema(schema_name, op_def):
    """ Find a schema, if any in the given op_def """

    # If nothing is supplied, check if a default schema
    # exists in the op attributes
    try:
        attr = op_def.attr[schema_name]
    except KeyError:
        return None
    else:
        if attr.type == "string":
            return attr.default_value.s

        return None


def get_tf_placeholders(op_def, call_args):
    """
    Get the tensorflow placeholder definitions derived from
    ``call_args`` and ``op_def``.

    Parameters
    ----------

    Returns
    -------
    dict of dict
        Dictionary containing the parameters required to create
        a placeholder for each input in ``call_args``.

        .. code-block::python

            {
                input_name: {
                    'allowed_types': [...],
                    'default_type_name': str,
                    'default': tf.dtype,
                    'schema': [dim1, dim2, ..., dimn]
                }
            }

    """
    fn = op_def.function
    fn_name = fn.__name__
    in_ph_info = []
    out_ph_info = []

    for input_name, input_def in op_def.inputs.items():
        arg = call_args[input_name]

        if arg is None:
            raise ValueError("Expected input '%s' to function '%s' was not "
                             "provided." % (input_name, fn_name))

        # Assume this is a normal variable for which
        # we don't need a placeholder
        if not isinstance(arg, mock.MagicMock):
            continue

        var_type = arg.var_type

        # Ignore, this is a known variable
        if var_type == KnownVariable:
            continue

        if var_type != UnknownVariable:
            continue
            raise ValueError("Input '%s' to function '%s' was not derived "
                             "from an established input (%s)"
                             % (input_name, fn_name, var_type))

        ph_name = arg.var_name

        arg_ph_info = arg_type_info(input_def, op_def)
        arg_ph_info.update({'dataset': arg.dataset, 'ops': set([fn_name])})

        # This input may have a dimension schema associated with it
        # which we can use to infer the shape
        schema_name = input_name + "_schema"

        try:
            # Try find something living in the kwargs
            schema = call_args[schema_name]
        except KeyError:
            schema = arg_schema(schema_name, op_def)

        if schema is not None:
            arg_ph_info['schema'] = parse_shape_schema(schema)

        # Assign the placeholder info for this argument
        in_ph_info.append((ph_name, arg_ph_info))

    for output_name, output_def in op_def.outputs.items():
        arg_ph_info = arg_type_info(output_def, op_def)
        arg_ph_info.update({'ops': set([fn_name])})

        schema = arg_schema(output_name + "_schema", op_def)

        if schema is not None:
            arg_ph_info['schema'] = schema

        out_ph_info.append((output_name, arg_ph_info))

    return in_ph_info, out_ph_info


def _while(cond, body, loop_vars, **kwargs):
    """
    Ensure that the condition and body of a tensorflow
    while_loop are invoked
    """

    print("tf.while_loop")
    cond(*loop_vars)
    return body(*loop_vars)


def _cond(pred, true_fn, false_fn, **kwargs):
    """
    Ensure that the predicate and both branches of the tensorflow
    conditional function are invoked
    """
    print("tf.cond")
    true_res = true_fn()
    false_res = false_fn()

    if pred():
        return true_res
    else:
        return false_res


def _case(pred_fn_pairs, *args, **kwargs):
    """
    Ensure that all predicates and functions of the tensorflow
    case statement are invoked
    """
    print("tf.case")
    ret = None

    for pred, fn in pred_fn_pairs:
        pred()
        val = fn()

        if ret is None:
            ret = val

    return ret


def _inspect_tf_op_call(*args, **kwargs):
    """
    Inspects call to a tensorflow operator

    Parameters
    ----------
    *args:
        operator arguments
    **kwargs:
        operator keyword arguments
    __op_def__ : tuple
        Tensorflow operator definition
    __op_placeholders__ : dict
        Existing placeholders
    """
    try:
        op_def = kwargs.pop("__op_def__")
    except KeyError:
        raise ValueError("__op_def__ not supplied")

    try:
        op_ph = kwargs.pop("__op_placeholders__")
    except KeyError:
        raise ValueError("__op_placeholders__ not supplied")

    # Generate the call arguments
    call_args = inspect.getcallargs(op_def.function, *args, **kwargs)

    # Find the missing placeholder definitions
    input_ph, output_ph = get_tf_placeholders(op_def, call_args)

    # Integrate missing into op placeholders,
    # checking against any existing values
    for k, new in input_ph:
        dataset = op_ph.setdefault(new.pop('dataset'), {})

        try:
            old = dataset[k]
        except KeyError:
            # Doesn't exist yet, assign and continue
            dataset[k] = new
            continue

        # Check that these attributes agree
        for attr in ('allowed_types', 'default', 'default_type_name'):
            if new[attr] != old[attr]:
                raise ValueError("input: '%s' "
                                 "old['%s']['%s'] (%s) != "
                                 "new['%s']['%s'] (%s)" %
                                 (k, attr, new[attr], k, attr, old[attr]))

        # We allow schema's to be optional
        new_schema = new.get('schema', None)
        old_schema = old.get('schema', None)

        # Take a new schema if we don't have an existing
        if old_schema is None and new_schema is not None:
            old['schema'] = new_schema
        # There is no new schema
        elif new_schema is None:
            pass
        # Old and new schema's should exist
        elif new_schema != old_schema:
            raise ValueError("input: '%s' "
                             "old['schema'] (%s) != new['schema'] (%s)" %
                             (k, old_schema, new_schema))

        # Add this op to the set of ops requiring this input placeholder
        old['ops'].update(new['ops'])

    # Create KnownVariable for each output
    outputs = tuple(mock.MagicMock(var_name=name,
                                   var_type=KnownVariable,
                                   var_info=info)
                    for name, info in output_ph)

    return outputs[0] if len(outputs) == 1 else outputs


MapDatasetInfo = namedtuple("MapDatasetInfo", ["placeholders", "tensor_map",
                                               "dataset", "map_keys",
                                               "put", "put_key", "close",
                                               "clear", "clear_key",
                                               "size"])


QueueDatasetInfo = namedtuple("QueueDatasetInfo", ["placeholders",
                                                   "tensor_queue", "dataset",
                                                   "put", "close", "size"])


def tensor_map(ds_name, ds_ph, dtypes, shapes):
    """
    Creates TensorMap dataset
    """
    tensor_map = TensorMap(dtypes, shapes, store=True)
    map_keys = tf.placeholder(tf.int64, shape=(None, 1),
                              name="%s_map_keys" % ds_name)
    put_key = tf.placeholder(tf.int64, shape=(),
                             name="%s_put_key" % ds_name)
    key_ds = tf.data.Dataset.from_tensor_slices(map_keys)
    map_dataset = MapDataset(key_ds, tensor_map, name=ds_name)
    put = tensor_map.insert(put_key, ds_ph)
    clear_keys = tf.placeholder(tf.int64, shape=(None,),
                                name="%s_clear_keys" % ds_name)
    clear = tensor_map.clear(clear_keys)

    close = tensor_map.close()
    size = tensor_map.size()

    return MapDatasetInfo(ds_ph, tensor_map, map_dataset,
                          map_keys, put, put_key, close,
                          clear, clear_keys, size)


def tensor_queue(ds_name, ds_ph, dtypes, shapes):
    """
    Creates TensorQueue dataset
    """
    tensor_queue = TensorQueue(dtypes, shapes)
    tensor_dataset = QueueDataset(tensor_queue, name=ds_name)
    put = tensor_queue.put(ds_ph)
    close = tensor_queue.close()
    size = tensor_queue.size()
    return QueueDatasetInfo(ds_ph, tensor_queue, tensor_dataset,
                            put, close, size)


def create_datasets(dataset_inputs, dataset_ph_info, ds_type="map"):
    """
    Creates datasets from inputs and placeholder info.

    If the type is ``map``, MapDatasets will be created,
    otherwise if the type is ``queue`` QueueDataset will be created.
    """

    _dims = {"(u,v,w)": 3, "(l,m)": 2, "(x,y,z)": 3, "corr": 4}
    dataset_info = {}

    # For each individual dataset
    for ds_name in dataset_inputs:
        # Get a dictionary holding the placeholders for this dataset
        ds_ph_info = dataset_ph_info[ds_name]
        inputs = dataset_inputs[ds_name]

        ds_ph = {}
        dtypes = {}
        shapes = {}

        # For each input
        for name in inputs.variables():
            # Try find existing placeholder information
            try:
                ph_info = ds_ph_info[name]
            except KeyError:
                # Handle internal '__<source>_keys__' inputs
                if not name.startswith("__") or not name.endswith("_keys__"):
                    raise ValueError("Unhandled input %s" % name)

                # Create placeholder for internal input
                dtypes[name] = dtype = tf.int64
                shapes[name] = shape = tf.TensorShape((None,))
                ds_ph[name] = tf.placeholder(dtype=dtype, shape=shape,
                                             name=name.lstrip("_"))
            else:

                # Create a placeholder for this input
                try:
                    dtype = ph_info['type']
                except KeyError:
                    raise ValueError("Type was not derived for input '%s' "
                                     "with definition '%s'" % (name, ph_info))

                try:
                    schema = ph_info['schema']
                except KeyError:
                    # No idea what kind of shape this tensor has
                    shape = tf.TensorShape(None)
                else:
                    shape = [d if isinstance(d, int) else _dims.get(d, None)
                             for d in schema]
                    shape = tf.TensorShape(shape)

                dtypes[name] = dtype
                shapes[name] = shape
                ds_ph[name] = tf.placeholder(dtype=dtype, shape=shape,
                                             name=name)

        if ds_type == "map":
            dataset_info[ds_name] = tensor_map(ds_name, ds_ph, dtypes, shapes)
        elif ds_type == "queue":
            dataset_info[ds_name] = tensor_queue(ds_name, ds_ph,
                                                 dtypes, shapes)
        else:
            raise ValueError("Wrong dataset type %s" % ds_type)

    return dataset_info


class VariableDict(dict):
    """
    Dictionary that creates :class:`mock.MagicMock` objects
    for missing dictionary entries.
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super(VariableDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        try:
            return super(VariableDict, self).__getitem__(key)
        except KeyError:
            pass

        data = mock.MagicMock(var_name=key, var_type=UnknownVariable,
                              dataset=self.name)
        super(VariableDict, self).__setitem__(key, data)
        return data


class FakeIterator(object):
    def __init__(self, name):
        self._var_dict = VariableDict(name)

    @property
    def initializer(self):
        return mock.MagicMock()

    def get_next(self):
        return self._var_dict


class FakeDataset(object):
    # Methods which return a dataset
    ds_methods = ['apply', 'batch', 'cache', 'concatenate', 'filter',
                  'flat_map', 'from_generator', 'from_sparse_tensor_slices',
                  'from_tensor_slices', 'from_tensors', 'interleave',
                  'list_files', 'map', 'padded_batch', 'prefetch', 'range',
                  'repeat', 'shard', 'shuffle', 'skip', 'take', 'zip']

    def __fake_dataset__(self, *args, **kwargs):
        return self

    def __init__(self, name):
        # TODO(sjperkins)
        # replace with metaclass
        for method in FakeDataset.ds_methods:
            setattr(self, method, self.__fake_dataset__)

        self._iterator = FakeIterator(name)

    def make_one_shot_iterator(self):
        return self._iterator

    def make_initializable_iterator(self):
        return self._iterator

    def variables(self):
        return self._iterator._var_dict


class DatasetsDict(dict):
    """
    Dictionary that creates :class:`VariableDict` objects
    for missing dictionary entries.
    """

    def __getitem__(self, key):
        try:
            return super(DatasetsDict, self).__getitem__(key)
        except KeyError:
            pass

        data = FakeDataset(key)
        super(DatasetsDict, self).__setitem__(key, data)
        return data


def FakeMapDataset(keys, tensor_map):
    return tensor_map.dataset


class FakeTensorMap(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset


class TensorMapDict(dict):
    """
    """
    def __init__(self, datasets):
        self._datasets = datasets

    def __getitem__(self, key):
        try:
            return super(TensorMapDict, self).__getitem__(key)
        except KeyError:
            pass

        data = FakeTensorMap(key, self._datasets[key])
        super(TensorMapDict, self).__setitem__(key, data)
        return data


def _float_types(cfg):
    try:
        precision = cfg['dtype']
    except KeyError:
        log.info("Defaulting to double floating point precision")
        precision = "double"

    if precision == "float":
        return {"FT": tf.float32, "CT": tf.complex64}
    elif precision == "double":
        return {"FT": tf.float64, "CT": tf.complex128}

    raise ValueError("Invalid precision %s" % precision)


def _set_placeholder_type(placeholder, float_types):
    ph = placeholder.copy()
    ph['type'] = float
    ph['type'] = float_types.get(ph['default_type_name'], ph['default'])
    return ph


def _set_placeholder_types(placeholders, float_types):
    return {ds: {n: _set_placeholder_type(ph, float_types)
                 for n, ph in ds_ph.items()}
            for ds, ds_ph in placeholders.items()}


def analyse_tensorflow_function(fn, cfg, device):
    """
    Finds the inputs and outputs required to feed tensorflow function ``fn``
    """
    mod = fn.__module__
    patch = mock.patch
    mocks = []

    # Mock the entire tensorflow module, as well as
    # the tensorflow control flow functions to ensure that
    # all their functions are called
    mocks.append(patch(".".join((mod, "tf"))))
    mocks.append(patch(".".join((mod, "tf.case")),
                       side_effect=_case))
    mocks.append(patch(".".join((mod, "tf.cond")),
                       side_effect=_cond))
    mocks.append(patch(".".join((mod, "tf.while_loop")),
                       side_effect=_while))
    mocks.append(patch(".".join((mod, "MapDataset")),
                       side_effect=FakeMapDataset))

    # Mock each RIME tensorflow function
    tfops_mod = "montblanc.impl.rime.tensorflow.tensorflow_ops"

    # Dictionary of placeholders created whenever a RIME tensorflow
    # function is called
    placeholders = {}

    for op_name, op_def in op_defs.items():
        target = ".".join((tfops_mod, op_def.function.__name__))
        # Curry def and placeholders into the side effect
        side_effect = partial(_inspect_tf_op_call,
                              __op_def__=op_def,
                              __op_placeholders__=placeholders)

        mocks.append(patch(target, side_effect=side_effect))

    # These objects fake Datasets and TensorMaps
    datasets = DatasetsDict()
    maps = TensorMapDict(datasets)

    device = tf.DeviceSpec(device)

    # Main input dataset
    input_ds = datasets["inputs"]

    with contextlib.nested(*mocks):
        out = fn(cfg, device, input_ds, maps)

    outputs = tuple((o.var_name, o.var_info) for o in out)

    for name, info in outputs:
        if "schema" not in info:
            raise ValueError("Schema is missing for output %s" % name)

    # Convert any floating point templates (FT, CT)
    # into concrete types
    float_types = _float_types(cfg)
    placeholders = _set_placeholder_types(placeholders, float_types)
    outputs = {n: _set_placeholder_type(ph, float_types)
               for n, ph in outputs}

    return datasets, placeholders, outputs
