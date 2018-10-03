from collections import namedtuple, OrderedDict
from os.path import join as pjoin
import re

import pkg_resources

import tensorflow as tf

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
    path_offset = pjoin('impl', 'rime', 'tensorflow', 'rime_ops')
    _rime_lib_path = pkg_resources.resource_filename("montblanc", path_offset)

_rime_so = tf.load_op_library(pjoin(_rime_lib_path, 'rime.so'))

__OP_TUPLE = namedtuple("__OP_TUPLE", ["inputs", "attr", "outputs",
                                       "orig_op_def", "function"])


def _xform_op_list(op_list):
    """
    Transform list-like protocol buffer representation
    into a more convenient dictionary rep
    """
    result = {}

    for op in op_list:
        snake_name = to_snake_case(op.name)
        result[snake_name] = __OP_TUPLE(
            OrderedDict((iarg.name, iarg) for iarg in op.input_arg),
            OrderedDict((attr.name, attr) for attr in op.attr),
            OrderedDict((oarg.name, oarg) for oarg in op.output_arg),
            op, getattr(_rime_so, snake_name))

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
        elif depth == 1 and schema[i] == ',':
            idx.append(i)

    idx.append(len(schema)-1)

    def _xform(substr):
        # Try integer conversion
        try:
            return int(substr)
        except ValueError:
            return substr

    return [_xform(schema[i+1:j].strip())
            for i, j in zip(idx, idx[1:])
            if i+1 != j]
