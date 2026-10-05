import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dptiny.data.fashion_mnist import get_fashion_mnist, CLASSES

# --- Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Load data (NOT flattened for CNN) ---
X_train, X_test, y_train, y_test = get_fashion_mnist(normalize=False, flatten=False)

# --- Standardize using training set stats ---
mean = X_train.mean()
std  = X_train.std()
X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

# --- Tensors ---
X_tr = torch.tensor(X_train)
y_tr = torch.tensor(y_train.astype(np.int64))
X_te = torch.tensor(X_test)
y_te = torch.tensor(y_test.astype(np.int64))

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_te, y_te), batch_size=64)

# --- Same CNN architecture as mnist_cnn.py ---
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32*7*7, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y_te)

# --- Train 10 epochs ---
for epoch in range(1, 11):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(xb)
        correct    += (pred.argmax(1) == yb).sum().item()
    train_acc = correct / len(y_tr)
    test_acc  = evaluate()
    print(f"Epoch {epoch:2d} | loss {total_loss/len(y_tr):.4f} | "
          f"train acc {train_acc:.4f} | test acc {test_acc:.4f}")

# --- Confusion matrix ---
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.extend(model(xb).argmax(1).tolist())
        all_true.extend(yb.tolist())

cm = np.zeros((10, 10), dtype=int)
for t, p in zip(all_true, all_pred):
    cm[t][p] += 1

print("\n10×10 Confusion Matrix (rows=true, cols=predicted):")
print("           " + " ".join(f"{c[:6]:>7}" for c in CLASSES))
for i, row in enumerate(cm):
    print(f"{CLASSES[i][:10]:>12} " + " ".join(f"{v:7d}" for v in row))

# Most confused pair
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
i, j = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
print(f"\nMost confused: '{CLASSES[i]}' predicted as '{CLASSES[j]}' ({cm_no_diag[i,j]} times)")

# Accuracy of each class
print("\nAccuracy of each class:")
for i, cls in enumerate(CLASSES):
    correct = cm[i][i]
    total = cm[i].sum()
    print(f"{cls:15s}: {correct/total:.4f}")