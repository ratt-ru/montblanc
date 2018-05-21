import ast
import inspect

import montblanc.impl.rime.tensorflow.tensorflow_ops as tf_ops
from montblanc.impl.rime.tensorflow.tensorflow_ops import op_defs

class WrapTensorflowCalls(ast.NodeTransformer):
    def __init__(self, fn_name):
        self._fn_name = fn_name

    def visit_FunctionDef(self, node):
        super(WrapTensorflowCalls, self).generic_visit(node)

        if node.name == self._fn_name:
            # Create tf_call_wrap import placed at top of function body
            tfops_imp = ast.ImportFrom(
                module='montblanc.impl.rime.tensorflow.tensorflow_ops',
                names=[ast.alias(name='tf_call_wrap', asname=None)],
                level=0)

            node =  ast.FunctionDef("capture_" + node.name,
                        node.args,
                        [tfops_imp] + node.body,
                        node.decorator_list)

        return node

    def visit_Call(self, node):
        super(WrapTensorflowCalls, self).generic_visit(node)

        if isinstance(node.func, ast.Name) and node.func.id in op_defs:
            node = ast.Call(func=ast.Name('tf_call_wrap', ast.Load()),
                args=[node.func]+node.args,
                keywords=node.keywords,
                starargs=node.starargs,
                kwargs=node.kwargs)

        elif isinstance(node.func, ast.Attribute) and node.func.attr in op_defs:
            node = ast.Call(func=ast.Name('tf_call_wrap', ast.Load()),
                args=[node.func]+node.args,
                keywords=node.keywords,
                starargs=node.starargs,
                kwargs=node.kwargs)

        return node

def rewrite_tensorflow_function(fn):
    fn_source = inspect.getsource(fn)
    tree = ast.parse(fn_source, filename="<generated-code>", mode="exec")
    tree = WrapTensorflowCalls(fn.__name__).visit(tree)
    tree = ast.fix_missing_locations(tree)
    code = compile(tree, filename="<generated-code>", mode="exec")
    exec(code)
    return locals()["capture_" + fn.__name__]


