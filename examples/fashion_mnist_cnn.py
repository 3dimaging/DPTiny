# imports: same as examples/mnist_cnn.py, plus your loader and CLASSES

# 1. seed xp.random before anything random happens
# 2. load with flatten=False
# 3. standardize: mean and std from the TRAINING set only, applied to both
# 4. model: copied unchanged from mnist_cnn.py
# 5. batch_size = 64, 10 epochs, Adam(lr=0.001)
# 6. each epoch: train loss (epoch average), train acc, test acc
#    (test inside test_mode() and no_grad())
# 7. after training: 10x10 confusion matrix (rows = true, cols = predicted),
#    per-class accuracy, most-confused pair (use CLASSES for names)
import time

import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
    xp,
)
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 42

try:
    if not is_available():
        raise RuntimeError("CuPy not installed")
    use_gpu()
    print("GPU enabled for training.")
except RuntimeError:
    print("GPU not available; training on CPU.")

# Seed after choosing the backend: xp is NumPy or CuPy depending on use_gpu(),
# and xp.random drives weight init, dropout masks and DataLoader shuffling.
xp.random.seed(SEED)
np.random.seed(SEED)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize with statistics from the TRAINING set only
mean = float(X_train.mean())
std = float(X_train.std())
X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)
print(f"train mean: {mean:.4f}, train std: {std:.4f}")

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

batch_size = 64
max_epoch = 10
data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

optimizer = Adam(model, lr=0.001)


def evaluate(loader):
    """Return (accuracy, predictions) on loader with dropout off."""
    preds = []
    sum_acc = 0.0
    count = 0
    with test_mode(), no_grad():
        for x, t in loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)
            sum_acc += float((pred == t).sum())
            count += len(t)
            preds.append(to_cpu(pred))
    return sum_acc / count, np.concatenate(preds)


start_time = time.time()
for epoch in range(max_epoch):
    sum_loss = 0.0
    sum_acc = 0.0
    count = 0
    model.train()
    for i, (x, t) in enumerate(data_loader):
        x = Variable(x)
        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        pred = y.data.argmax(axis=1)
        sum_loss += float(loss.data) * len(t)
        sum_acc += float((pred == t).sum())
        count += len(t)

    train_loss = sum_loss / count
    train_acc = sum_acc / count
    test_acc, _ = evaluate(test_loader)
    print(
        f"epoch: {epoch + 1:2d}, train loss: {train_loss:.4f}, "
        f"train acc: {train_acc:.4f}, test acc: {test_acc:.4f}, "
        f"time: {time.time() - start_time:.2f}s"
    )

print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
print("(Train accuracy is measured during training, with dropout active.)")

# ---- Final analysis ---------------------------------------------------------

# Predict every test image once more with dropout off
test_acc, preds = evaluate(test_loader)
# Move labels back to the CPU (CuPy arrays on GPU) for NumPy analysis
true = to_cpu(y_test).astype(np.int64)
preds = preds.astype(np.int64)
print(f"Final test accuracy: {test_acc:.4f}")

# Confusion matrix: 
# rows = true class, columns = predicted class.
# cm[i, j] = number of test images of class i that were predicted as class j.
# Each (true, pred) pair is encoded as one number
# reshape(10, 10) puts
# count number true * 10 + pred back at row true, column pred
num_classes = len(CLASSES)
cm = np.bincount(
    true * num_classes + preds, minlength=num_classes * num_classes
).reshape(num_classes, num_classes)

# Print the matrix with full class names; column width fits the longest name.
short = list(CLASSES)
width = max(len(c) for c in CLASSES) + 2
print("\nConfusion matrix (rows = true, cols = predicted):")
print(" " * 12 + "".join(f"{s:>{width}}" for s in short))
for i in range(num_classes):
    print(f"{CLASSES[i]:>12}" + "".join(f"{v:{width}d}" for v in cm[i]))

# Per-class accuracy: 
# correct predictions (diagonal) divided by the number of test images in that class
print("\nPer-class accuracy:")
per_class = cm.diagonal() / cm.sum(axis=1)
for name, acc in zip(CLASSES, per_class):
    print(f"{name:>12}: {acc:.4f}")

# Most-confused pair, one direction: 
# largest off-diagonal entry.
off = cm.copy()
np.fill_diagonal(off, 0)
i, j = np.unravel_index(off.argmax(), off.shape)
print(
    f"\nMost confused (one direction): true {CLASSES[i]} predicted as "
    f"{CLASSES[j]} ({off[i, j]} images)"
)

# Most-confused pair, both directions combined: 
# (off + off.T)[i, j] counts i predicted as j plus j predicted as i
sym = np.triu(off + off.T, k=1)
a, b = np.unravel_index(sym.argmax(), sym.shape)
print(
    f"Most confused (both directions): {CLASSES[a]} <-> {CLASSES[b]} "
    f"({sym[a, b]} images: {off[a, b]} + {off[b, a]})"
)
