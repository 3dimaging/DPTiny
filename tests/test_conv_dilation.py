import numpy as np
import pytest

from dptiny import Variable
from dptiny import functions as F
from dptiny.nn import Conv2d
from dptiny.utils import gradient_check


@pytest.mark.parametrize('stride,pad,dilation', [
    (1, 0, 2), ((2, 1), (1, 2), (2, 3)), (1, 1, 1),
])
def test_dilated_convolution(stride, pad, dilation):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1, 2, 5, 7))
    w = rng.normal(size=(2, 2, 2, 2))
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    oh = (5 + 2 * ph - dh - 1) // sh + 1
    ow = (7 + 2 * pw - dw - 1) // sw + 1
    expected = np.zeros((1, 2, oh, ow))
    for h in range(oh):
        for v in range(ow):
            patch = padded[:, :, h*sh:h*sh+dh+1:dh, v*sw:v*sw+dw+1:dw]
            expected[:, :, h, v] = np.einsum('nchw,ochw->no', patch, w)
    f = lambda a, b: F.conv2d(a, b, stride, pad, dilation)
    np.testing.assert_allclose(f(x, w).data, expected, atol=1e-12)
    assert gradient_check(f, x, w)


@pytest.mark.parametrize('kwargs', [
    {'stride': 0}, {'stride': (1, -1)}, {'pad': -1},
    {'pad': (0, -1)}, {'dilation': 0}, {'dilation': (1, -2)},
    {'dilation': 8},
])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        F.conv2d(np.ones((1, 1, 3, 3)), np.ones((1, 1, 2, 2)), **kwargs)


def test_layer_dilation_and_dtype():
    layer = Conv2d(2, 3, 3, 1, 2, 2, False, 'he', np.float64)
    x = Variable(np.ones((1, 2, 5, 5)))
    y = layer(x)
    assert y.shape == (1, 3, 5, 5)
    assert layer.W.dtype == layer.b.dtype == np.float64
    y.sum().backward()
    assert x.grad.shape == x.shape
    assert layer.W.grad.shape == layer.W.shape


def test_dilation_one_backward_compatibility():
    rng = np.random.default_rng(123)
    x1 = Variable(rng.normal(size=(1, 2, 5, 5)))
    x2 = Variable(x1.data.copy())
    w1 = Variable(rng.normal(size=(3, 2, 3, 3)))
    w2 = Variable(w1.data.copy())
    y_default = F.conv2d(x1, w1, stride=1, pad=1)
    y_dilation1 = F.conv2d(x2, w2, stride=1, pad=1, dilation=1)
    np.testing.assert_allclose(
        y_default.data, y_dilation1.data, atol=1e-12,
    )
    y_default.sum().backward()
    y_dilation1.sum().backward()
    np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-12)
    np.testing.assert_allclose(w1.grad, w2.grad, atol=1e-12)