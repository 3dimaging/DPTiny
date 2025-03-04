import numpy as np
from dptiny.core import Variable
from dptiny.core import Config
from dptiny.core import no_grad
from dptiny.core import as_array, as_variable
from dptiny.functions import (
    Function,
    add, mul, neg, sub, div, rsub, rdiv,
    sigmoid, relu, softmax, softmax_cross_entropy,
    matmul
)
from dptiny.layers import Linear, MLP, SGD
from dptiny.datasets import get_mnist, DataLoader

__version__ = '0.1.0'

# Define special methods for Variable class to enable operator overloading
def _add(x0, x1):
    return add(x0, x1)

def _radd(x0, x1):
    return add(x1, x0)

def _mul(x0, x1):
    return mul(x0, x1)

def _rmul(x0, x1):
    return mul(x1, x0)

def _neg(x):
    return neg(x)

def _sub(x0, x1):
    return sub(x0, x1)

def _rsub(x0, x1):
    return sub(x1, x0)

def _truediv(x0, x1):
    return div(x0, x1)

def _rtruediv(x0, x1):
    return div(x1, x0)

Variable.__add__ = _add
Variable.__radd__ = _radd
Variable.__mul__ = _mul
Variable.__rmul__ = _rmul
Variable.__neg__ = _neg
Variable.__sub__ = _sub
Variable.__rsub__ = _rsub
Variable.__truediv__ = _truediv
Variable.__rtruediv__ = _rtruediv
