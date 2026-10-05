import numpy as np

from dptiny import Variable, no_grad, softmax_cross_entropy, test_mode
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import Conv2d, Dropout, Flatten, Linear, MaxPool2d, ReLU, Sequential
from dptiny.optim import Adam

# Fix the random seed so every run gives the same result
np.random.seed(42)

# ---------- Load data ----------
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

# Standardize using the mean and std of the TRAINING set only
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
print("train mean:", round(float(mean), 4), " train std:", round(float(std), 4))

# ---------- Model (same as examples/mnist_cnn.py) ----------
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

train_loader = DataLoader((X_train, y_train), 64)
test_loader = DataLoader((X_test, y_test), 64, shuffle=False)
optimizer = Adam(model, lr=0.001)


def get_predictions(loader):
    """Run the model on a dataset and return true labels and predictions."""
    all_true = []
    all_pred = []
    with test_mode(), no_grad():
        for x, t in loader:
            y = model(Variable(x))
            all_pred.append(y.data.argmax(axis=1))
            all_true.append(t)
    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------- Training ----------
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    seen = 0

    for x, t in train_loader:
        y = model(Variable(x))
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        total_loss += float(loss.data) * len(t)
        correct += (y.data.argmax(axis=1) == t).sum()
        seen += len(t)

    true, pred = get_predictions(test_loader)
    test_acc = (true == pred).mean()

    print(f"Epoch {epoch + 1}: train loss {total_loss / seen:.4f}, "
          f"train acc {correct / seen:.4f}, test acc {test_acc:.4f}")

# ---------- Final evaluation ----------
true, pred = get_predictions(test_loader)

# Confusion matrix: row = real class, column = predicted class
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(true, pred):
    cm[t, p] += 1

print("\nConfusion matrix (row = real, column = predicted):")
print(cm)

print("\nAccuracy of each class:")
for i in range(10):
    print(f"  {CLASSES[i]:12s} {cm[i, i] / cm[i].sum():.4f}")

# Find the two classes that get mixed up the most (both directions added)
worst_count = 0
worst_pair = None
for i in range(10):
    for j in range(i + 1, 10):
        mistakes = cm[i, j] + cm[j, i]
        if mistakes > worst_count:
            worst_count = mistakes
            worst_pair = (i, j)

a, b = worst_pair
print(f"\nMost confused pair: {CLASSES[a]} and {CLASSES[b]} "
      f"({worst_count} mistakes)")
