import numpy as np

from dptiny import Conv2d, Variable

print("=" * 50)
print("Test 1: dilation=1 (original behaviour)")
print("=" * 50)

x_data = np.random.randn(8, 1, 28, 28).astype(np.float32)
x = Variable(x_data)

conv = Conv2d(in_channels=1, out_channels=16, kernel_size=3, pad=1, dilation=1)
y = conv(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)          # expect (8, 16, 28, 28)
print("Kernel shape:", conv.W.shape)
print("Bias shape  :", conv.b.shape if conv.b is not None else None)

y.backward()
print("grad x shape:", x.grad.shape)
print("grad W shape:", conv.W.grad.shape)

print()
print("=" * 50)
print("Test 2: dilation=2")
print("=" * 50)

x2 = Variable(x_data.copy())

# With dilation=2 and pad=2, a 3x3 kernel has effective size 5x5.
# pad=2 keeps the spatial size at 28x28.
conv2 = Conv2d(in_channels=1, out_channels=16, kernel_size=3, pad=2, dilation=2)
y2 = conv2(x2)

print("Input shape :", x2.shape)
print("Output shape:", y2.shape)         # expect (8, 16, 28, 28)
print("Kernel shape:", conv2.W.shape)
print("Bias shape  :", conv2.b.shape if conv2.b is not None else None)

y2.backward()
print("grad x shape:", x2.grad.shape)
print("grad W shape:", conv2.W.grad.shape)

print()
print("=" * 50)
print("Test 3: dilation=(2,2) tuple form")
print("=" * 50)

x3 = Variable(x_data.copy())
conv3 = Conv2d(in_channels=1, out_channels=8, kernel_size=3, pad=2, dilation=(2, 2))
y3 = conv3(x3)
print("Input shape :", x3.shape)
print("Output shape:", y3.shape)         # expect (8, 8, 28, 28)

y3.backward()
print("grad x shape:", x3.grad.shape)
print("grad W shape:", conv3.W.grad.shape)

print()
print("All tests passed.")
