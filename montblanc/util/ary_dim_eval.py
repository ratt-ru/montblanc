# Based on http://stackoverflow.com/a/9558001

import ast
import operator as op

# supported operators
operators = { ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
    ast.USub: op.neg}

def eval_expr(expr, variables=None):
    """
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    if variables is None: variables = {}

    def __eval(node):
        if isinstance(node, ast.Num): # <number>
            return node.n
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            return operators[type(node.op)](
                __eval(node.left),
                __eval(node.right))
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            return operators[type(node.op)](
                __eval(node.operand))
        elif isinstance(node, ast.Name):
            try:
                return variables[node.id]
            except KeyError:
                raise ValueError, ('Cannot find a matching variable for '
                    'parse tree name %s') % node.id
        else:
            raise TypeError(node)

    return __eval(ast.parse(expr, mode='eval').body)


def eval_expr_names_and_nrs(expr):
    def __eval(node, l):
        if isinstance(node, ast.Num): # <number>
            l.append(node.n)
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            __eval(node.left, l)
            __eval(node.right, l)
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            pass # ignore this
        elif isinstance(node, ast.Name):
            l.append(node.id)
        else:
            raise TypeError(node)

    l = []

    __eval(ast.parse(expr, mode='eval').body, l)
    return l