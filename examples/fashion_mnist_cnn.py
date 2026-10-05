
import time
import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)

from dptiny.backend import xp, to_cpu
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import get_fashion_mnist, CLASSES

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


# --------------------------------------------------
# GPU / CPU
# --------------------------------------------------

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")


# --------------------------------------------------
# Random seed
# --------------------------------------------------

SEED = 42
np.random.seed(SEED)
xp.random.seed(SEED)


# --------------------------------------------------
# Load Fashion-MNIST
# --------------------------------------------------

print("Loading Fashion-MNIST dataset...")

X_train, X_test, y_train, y_test = get_fashion_mnist(
    flatten=False
)


# --------------------------------------------------
# Standardization using TRAINING data only
# --------------------------------------------------

train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print(f"Training mean: {train_mean:.6f}")
print(f"Training std: {train_std:.6f}")


# Move data to GPU if available
if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)


# --------------------------------------------------
# CNN - same architecture as mnist_cnn.py
# --------------------------------------------------

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

data_loader = DataLoader(
    (X_train, y_train),
    batch_size
)

test_loader = DataLoader(
    (X_test, y_test),
    batch_size,
    shuffle=False
)

optimizer = Adam(model, lr=0.001)

start_time = time.time()




for epoch in range(max_epoch):

    train_loss = 0.0
    train_correct = 0
    train_count = 0

    model.train()

    for x, t in data_loader:

        x = Variable(x)

        y = model(x)

        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        n = len(t)

        train_loss += float(loss.data) * n

        pred = y.data.argmax(axis=1)

        train_correct += int((pred == t).sum())
        train_count += n


    avg_train_loss = train_loss / train_count

    train_accuracy = train_correct / train_count


   

    test_correct = 0
    test_count = 0

    with test_mode(), no_grad():

        for x, t in test_loader:

            y = model(Variable(x))

            pred = y.data.argmax(axis=1)

            test_correct += int((pred == t).sum())
            test_count += len(t)


    test_accuracy = test_correct / test_count


    print(
        f"Epoch {epoch + 1}/10 | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Test Accuracy: {test_accuracy:.4f}"
    )


print(
    f"\nTraining completed in "
    f"{time.time() - start_time:.2f} seconds"
)




confusion_matrix = np.zeros(
    (10, 10),
    dtype=np.int64
)

with test_mode(), no_grad():

    for x, t in test_loader:

        y = model(Variable(x))

        pred = y.data.argmax(axis=1)

        pred = np.asarray(to_cpu(pred))
        true_labels = np.asarray(to_cpu(t))

        np.add.at(
            confusion_matrix,
            (true_labels, pred),
            1
        )


print("\nConfusion Matrix:")
print(confusion_matrix)




print("\nAccuracy by class:")

for i, class_name in enumerate(CLASSES):

    total = confusion_matrix[i].sum()

    correct = confusion_matrix[i, i]

    accuracy = correct / total

    print(
        f"{class_name}: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )




max_confusion = -1
most_confused_pair = None

for i in range(10):

    for j in range(i + 1, 10):

        confusion = (
            confusion_matrix[i, j]
            + confusion_matrix[j, i]
        )

        if confusion > max_confusion:

            max_confusion = confusion

            most_confused_pair = (i, j)


i, j = most_confused_pair

print("\nMost confused classes:")

print(
    f"{CLASSES[i]} and {CLASSES[j]} "
    f"({max_confusion} total confusions)"
)
