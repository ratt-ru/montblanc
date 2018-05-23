from __future__ import print_function

import ast
import inspect

try:
    from cytoolz import merge
except ImportError:
    from toolz import merge

import montblanc.impl.rime.tensorflow.tensorflow_ops as tf_ops
from montblanc.impl.rime.tensorflow.tensorflow_ops import (op_defs,
                                                           parse_shape_schema)

import tensorflow as tf


def ast_dump(node, annotate_fields=True, include_attributes=False, indent='  '):
    """
    Return a formatted dump of the tree in *node*.  This is mainly useful for
    debugging purposes.  The returned string will show the names and the values
    for fields.  This makes the code impossible to evaluate, so if evaluation is
    wanted *annotate_fields* must be set to False.  Attributes such as line
    numbers and column offsets are not dumped by default.  If this is wanted,
    *include_attributes* can be set to True.
    """
    def _format(node, level=0):
        if isinstance(node, ast.AST):
            fields = [(a, _format(b, level)) for a, b in ast.iter_fields(node)]
            if include_attributes and node._attributes:
                fields.extend([(a, _format(getattr(node, a), level))
                               for a in node._attributes])
            return ''.join([
                node.__class__.__name__,
                '(',
                ', '.join(('%s=%s' % field for field in fields)
                           if annotate_fields else
                           (b for a, b in fields)),
                ')'])
        elif isinstance(node, list):
            lines = ['[']
            lines.extend((indent * (level + 2) + _format(x, level + 2) + ','
                         for x in node))
            if len(lines) > 1:
                lines.append(indent * (level + 1) + ']')
            else:
                lines[-1] += ']'
            return '\n'.join(lines)
        return repr(node)

    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _format(node)


def get_tf_placeholders(op_def, args, kwargs):
    arg_spec = inspect.getargspec(op_def.function)

    # tensorflow doesn't seem to generate varargs, keywords or
    # (actual) defaults for custom operator python bindings.
    # fail in anticipation of properly handling these,
    # if they are introduced
    if arg_spec.varargs is not None:
        raise ValueError("Unhandled *args")

    if arg_spec.keywords is not None:
        raise ValueError("Unhandled *kwargs")

    if (arg_spec.defaults is not None and
            any(a is not None for a in arg_spec.defaults)):
        raise ValueError("Unhandled defaults")


    ph_info = {}

    # Convert list of ast.keyword objects to dict
    kwargs = {kw.arg: kw.value for kw in kwargs}

    for name, input_def in op_def.inputs.items():
        # Get the ast arg definition
        arg = args.pop(0)

        if (isinstance(arg, ast.Subscript) and
            isinstance(arg.slice.value, ast.Str) and
            arg.value.id.endswith("inputs")):

            # Get the string value of the slice
            ph_name = arg.slice.value.s

            if input_def.type:
                # Fixed type, easy
                dtype = tf.as_dtype(input_def.type)
                type_name = dtype.name
                allowed = [dtype]
            elif input_def.type_attr:
                # If a polymorphic type, there'll be an attribute
                # with a default type associated
                type_name = input_def.type_attr
                type_attr = op_def.attr[input_def.type_attr]
                allowed = type_attr.allowed_values.list
                allowed = [tf.as_dtype(dt) for dt in allowed.type]
                dtype = tf.as_dtype(type_attr.default_value.type)
            elif input_def.type_list_attr:
                # Implement me
                raise ValueError("Type Lists not handled")
            else:
                raise TypeError("Couldn't infer type "
                                "of missing input %s" % name)

            arg_ph_info = {
                'allowed_types': allowed,
                'default_type_name': type_name,
                'default': dtype,
            }

            # This input may have a dimension schema associated with it
            # which we can use to infer the shape
            schema_name = name + "_schema"

            try:
                # Try find something living in the kwargs
                ast_schema = kwargs[schema_name]
            except KeyError:
                # Check if a default schema is living in the
                # op schemas
                try:
                    attr = op_def.attr[schema_name]
                    if attr.type == "string":
                        schema = attr.default_value.s
                    else:
                        schema = None
                except KeyError:
                    schema = None
            else:
                if isinstance(ast_schema, ast.Str):
                    schema = ast_schema.s

            if schema is not None:
                arg_ph_info['schema'] = parse_shape_schema(schema)

            ph_info[ph_name] = arg_ph_info

    return ph_info

class TensorflowGraphAnalyser(ast.NodeVisitor):
    def __init__(self, fn):
        self._fn_name = fn.__name__
        self._in_fn_call = None

    def visit_FunctionDef(self, node):
        if node.name == self._fn_name:
            pass
            #print("Entered", node.name)

        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            raise TypeError("Unhandled ast type %r in visit_Call" % type(node.func))
        try:
            op_def = op_defs[func_name]
        except KeyError:
            self.generic_visit(node)
            return

        from pprint import pprint
        kwargs = get_tf_placeholders(op_def, node.args, node.keywords)
        pprint([func_name, kwargs])

        self._in_fn_call = func_name
        self.generic_visit(node)
        self._in_fn_call = None

    def visit_Subscript(self, node):
        if (self._in_fn_call is None and
            isinstance(node.value, ast.Name) and
            node.value.id.endswith("inputs") and
            isinstance(node.slice.value, ast.Str)):

            print("INPUT %s[%s]" % (node.value.id, node.slice.value.s))

        self.generic_visit(node)

    def visit_Assign(self, node):
        #print(ast_dump(node))
        self.generic_visit(node)

def analyse_tensorflow_function(fn):
    fn_source = inspect.getsource(fn)
    tree = ast.parse(fn_source, filename="<ast>", mode="exec")

    analyser = TensorflowGraphAnalyser(fn)
    analyser.visit(tree)


