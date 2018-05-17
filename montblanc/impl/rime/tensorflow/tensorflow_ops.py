import inspect
from collections import namedtuple, OrderedDict
from os.path import join as pjoin
import re

import pkg_resources

import tensorflow as tf
from tensorflow.python.framework.dtypes import as_dtype

# Convert tensorflow CamelCase op names to python snake case
_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z0-9])([A-Z])')

def to_snake_case(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()

# Load standard/development version of rime tensorflow library?
if True:
    # Installed library location
    _rime_lib_path = pkg_resources.resource_filename("montblanc", "ext")
else:
    # Development library location
    _rime_lib_path = pkg_resources.resource_filename("montblanc",
                            pjoin('impl', 'rime', 'tensorflow', 'rime_ops'))

_rime_so = tf.load_op_library(pjoin(_rime_lib_path, 'rime.so'))

__OP_TUPLE = namedtuple("__OP_TUPLE", ["inputs", "outputs", "attr", "orig_op"])

def _xform_op_list(op_list):
    """
    Transform list-like protocol buffer representation
    into a more convenient dictionary rep
    """
    result = {}

    for op in op_list:
        result[to_snake_case(op.name)] = __OP_TUPLE(
            OrderedDict((iarg.name, iarg) for iarg in op.input_arg),
            OrderedDict((oarg.name, oarg) for oarg in op.output_arg),
            OrderedDict((attr.name, attr) for attr in op.attr),
            op)

    return result

# Export operators into the namespace of this module
op_defs = _xform_op_list(_rime_so.OP_LIST.op)
globals().update({n: getattr(_rime_so, n) for n in op_defs.keys()})

def parse_shape_schema(schema):
    idx = []
    depth = 1

    if schema[0] != '(' or schema[-1] != ')':
        raise ValueError("schema must be surrounded by parenthesis")

    idx.append(0)

    for i in range(1, len(schema) - 1):
        if schema[i] == '(':
            depth += 1
        elif schema[i] == ')':
            depth -= 1
        elif depth ==1 and schema[i] == ',':
            idx.append(i)

    idx.append(len(schema)-1)

    return [schema[i+1:j] for i, j in zip(idx, idx[1:]) if i+1 != j]

def tf_call_wrap(fn, *args, **kwargs):
    arg_spec = inspect.getargspec(fn)

    # tensorflow doesn't seem to generate varargs, keywords or
    # (actual) defaults for custom operator python bindings.
    # fail in anticipation of properly handling these,
    # if they are introduced
    if not arg_spec.varargs is None:
        raise ValueError("Unhandled *args")

    if not arg_spec.keywords is None:
        raise ValueError("Unhandled *kwargs")

    if (arg_spec.defaults is not None and
            any(a is not None for a in arg_spec.defaults)):
        raise ValueError("Unhandled defaults")

    op_def = op_defs[fn.__name__]
    fn_kwargs = {name: val for name, val in zip(arg_spec.args, args)}

    # Handle any remaining arguments
    for name in arg_spec.args[len(args):]:
        if name == "name":
            continue
        # Handle input arguments
        elif name in op_def.inputs:
            try:
                # Try get input from the user
                fn_kwargs[name] = kwargs[name]
            except KeyError:
                # We have no input, we should create a placeholder for it...
                input_spec = op_def.inputs[name]

                # Fixed type, easy
                if input_spec.type:
                    arg_type = input_spec.type
                # If a polymorphic type, there'll be an attribute
                # with a default type associated
                elif input_spec.type_attr:
                    type_attr = op_def.attrs[input_spec.type_attr]
                    dtype = type_attr.default_value.type
                else:
                    raise TypeError("Couldn't infer type "
                                    "of missing input %s" % name)

                # Convert to a tensorflow dtype
                dtype = as_dtype(arg_type)

                # This input may have a dimension schema associated with it
                # which we can use to infer the shape
                schema = op_def.attr.get(name + "_schema", None)

                if schema is not None:
                    shape = tf.TensorShape(*(None for d in len(schema)))
                else:
                    shape = tf.TensorShape(None)

                # Create the placeholder
                fn_kwargs[name] = tf.placeholder(arg_type, shape)

        # Handle Attributes
        elif name in op_def.attr:
            try:
                fn_kwargs[name] = kwargs[name]
            except KeyError:
                pass
        else:
            raise ValueError("Unable to set arg=%s" % name)

    return fn(**fn_kwargs)
