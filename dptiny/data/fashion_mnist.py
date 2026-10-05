import numpy as np
import os
import gzip
import urllib.request

CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def _download(data_home):
    os.makedirs(data_home, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(data_home, fname)
        if not os.path.exists(dest):
            tmp = dest + ".tmp"
            try:
                urllib.request.urlretrieve(BASE_URL + fname, tmp)
                os.rename(tmp, dest)  # atomic rename — no half-written files
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise

def _parse_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Header: magic (4 bytes), then dims (each 4 bytes)
    magic = int.from_bytes(data[0:4], "big")
    dtype_code = (magic >> 8) & 0xFF
    ndims = magic & 0xFF
    if dtype_code != 0x08:
        raise ValueError(f"Expected uint8 IDX file, got dtype code {dtype_code:#x}")
    dims = [int.from_bytes(data[4 + 4*i : 8 + 4*i], "big") for i in range(ndims)]
    arr = np.frombuffer(data[4 + 4*ndims:], dtype=np.uint8).reshape(dims)
    return arr

def get_fashion_mnist(normalize=True, flatten=True, data_home=None):
    if data_home is None:
        data_home = os.path.expanduser("~/.cache/dptiny")
    _download(data_home)

    X_train = _parse_idx(os.path.join(data_home, "train-images-idx3-ubyte.gz"))
    y_train = _parse_idx(os.path.join(data_home, "train-labels-idx1-ubyte.gz"))
    X_test  = _parse_idx(os.path.join(data_home, "t10k-images-idx3-ubyte.gz"))
    y_test  = _parse_idx(os.path.join(data_home, "t10k-labels-idx1-ubyte.gz"))

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test  /= 255.0

    if flatten:
        X_train = X_train.reshape(len(X_train), 784)
        X_test  = X_test.reshape(len(X_test), 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test  = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_fashion_mnist()
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"X_train dtype: {X_train.dtype}")
    print(f"y_train dtype: {y_train.dtype}")
    print(f"Classes: {CLASSES}")