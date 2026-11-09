import numpy as np

from dptiny import Conv2d, Variable

# Dilation factor used by this example. Set to 1 for an ordinary convolution.
DILATION = 2

# Random 8 images of 1x28x28 (MNIST-like shape: NCHW)
x_data = np.random.randn(8, 1, 28, 28).astype(np.float32)
x = Variable(x_data)

# A dilated KxK kernel covers an effective window of D*(K-1)+1 pixels, so
# padding by D keeps the spatial size unchanged for a 3x3 kernel.
kernel_size = 3
effective = DILATION * (kernel_size - 1) + 1

# Conv layer: 1 -> 16 channels, 3x3 kernel, dilated, padded to keep the size
conv = Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=kernel_size,
    pad=DILATION,
    dilation=DILATION,
)
y = conv(x)

print("Dilation:", DILATION)
print("Effective kernel size:", f"{effective}x{effective}")
print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Kernel shape:", conv.W.shape)
print("Bias shape:", conv.b.shape if conv.b is not None else None)

# Backward
y.backward()
print("Gradient of input shape:", x.grad.shape)
print("Gradient of kernel shape:", conv.W.grad.shape)

# The parameter count does not change with dilation: the same 3x3 weights
# simply look at a wider area of the input.
print("Parameter count:", conv.W.data.size + (conv.b.data.size if conv.b else 0))
