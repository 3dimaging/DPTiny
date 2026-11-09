"""Tests for dilated (atrous) 2-D convolution."""

import numpy as np
import pytest

import dptiny.functions as F
from dptiny import Variable
from dptiny.nn import Conv2d
from dptiny.utils import gradient_check


@pytest.fixture(autouse=True)
def _keep_global_rng():
    """Do not shift NumPy's global RNG stream used by other test files."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _dilate_kernel(W, dilation):
    """Insert (d - 1) zeros between kernel taps: a reference for dilation."""
    DH, DW = dilation
    OC, C, KH, KW = W.shape
    Wd = np.zeros((OC, C, DH * (KH - 1) + 1, DW * (KW - 1) + 1), dtype=W.dtype)
    Wd[:, :, ::DH, ::DW] = W
    return Wd


CASES = [
    # stride, pad, dilation, kernel (KH, KW), input (H, W)
    (1, 0, 2, (3, 3), (9, 9)),
    (1, 2, 2, (3, 3), (8, 8)),
    (2, 1, (2, 3), (3, 3), (11, 12)),
    ((2, 1), (0, 3), (3, 1), (3, 2), (10, 9)),
    (1, 1, 4, (3, 3), (9, 10)),
    (3, 2, (1, 2), (2, 3), (12, 12)),
]


@pytest.mark.parametrize("stride,pad,dilation,ksize,hw", CASES)
def test_dilated_forward_matches_zero_inserted_kernel(stride, pad, dilation, ksize, hw):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3) + hw)
    W = rng.standard_normal((4, 3) + ksize)
    d = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    y = F.conv2d(Variable(x), Variable(W), stride=stride, pad=pad, dilation=dilation)
    Wd = Variable(_dilate_kernel(W, d))
    y_ref = F.conv2d(Variable(x), Wd, stride=stride, pad=pad)
    assert y.shape == y_ref.shape
    assert np.allclose(y.data, y_ref.data)


@pytest.mark.parametrize("stride,pad,dilation,ksize,hw", CASES)
def test_dilated_gradcheck(stride, pad, dilation, ksize, hw):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 2) + hw)
    W = rng.standard_normal((3, 2) + ksize)
    assert gradient_check(
        lambda a, w: F.conv2d(a, w, stride=stride, pad=pad, dilation=dilation), x, W
    )


def test_output_shape_formula():
    x = Variable(np.zeros((1, 1, 28, 28)))
    W = Variable(np.zeros((1, 1, 3, 3)))
    # K_eff = 2 * (3 - 1) + 1 = 5 -> (28 + 4 - 5) // 1 + 1 = 28
    assert F.conv2d(x, W, pad=2, dilation=2).shape == (1, 1, 28, 28)
    # K_eff = 9 -> (28 - 9) // 2 + 1 = 10
    assert F.conv2d(x, W, stride=2, dilation=4).shape == (1, 1, 10, 10)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stride=0),
        dict(stride=(1, -1)),
        dict(pad=-1),
        dict(dilation=0),
        dict(dilation=(1, 0)),
        dict(dilation=3),  # K_eff = 7 > 5
        dict(dilation=(3, 1), pad=(0, 1)),  # KH_eff = 7 > 5
    ],
)
def test_invalid_arguments_raise(kwargs):
    x = Variable(np.ones((1, 1, 5, 5)))
    W = Variable(np.ones((1, 1, 3, 3)))
    with pytest.raises(ValueError):
        F.conv2d(x, W, **kwargs)


def test_effective_kernel_exactly_fits():
    x = Variable(np.ones((1, 1, 5, 5)))
    W = Variable(np.ones((1, 1, 3, 3)))
    assert F.conv2d(x, W, dilation=2).shape == (1, 1, 1, 1)


def test_layer_positional_order_and_forward():
    # in_channels, out_channels, kernel_size, stride, pad, dilation
    conv = Conv2d(3, 8, 3, 1, 2, 2)
    assert conv.dilation == 2
    x = Variable(np.random.randn(2, 3, 16, 16).astype(np.float32))
    y = conv(x)
    assert y.shape == (2, 8, 16, 16)
    y.sum().backward()
    assert conv.W.grad.shape == conv.W.shape


def test_layer_old_callers_unchanged():
    conv = Conv2d(1, 4, 3, pad=1)
    assert conv.dilation == 1
    assert conv(Variable(np.random.randn(1, 1, 6, 6))).shape == (1, 4, 6, 6)
    conv = Conv2d(1, 4, 3, 1, 1, nobias=True)
    assert conv.b is None
