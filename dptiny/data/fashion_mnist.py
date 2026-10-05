"""Fashion-MNIST dataset loader."""

import gzip
import os
import urllib.request

import numpy as np

CLASSES = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
            "Shirt", "Sneaker", "Bag", "Ankle boot")

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")

_BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels":  "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

def _download(url, path):
    """Download url to path without ever leaving a partial file at path."""
    # write to a temp file in the same folder
    # os.replace() it into place only after it finishes
    # delete the temp file if anything goes wrong
    path_name = path + ".part"
    try:
        urllib.request.urlretrieve(url, path_name)
        os.replace(path_name, path)
    except BaseException:
        if os.path.exists(path_name):
            os.remove(path_name)
        raise


def _parse_idx(data):
    """Parse raw (already un-gzipped) IDX bytes into a NumPy array."""
    # check bytes 0-1 are zero, byte 2 is 0x08, read ndim from byte 3
    # read the ndim sizes as big-endian uint32
    # check total length == 4 + 4*ndim + product(sizes)
    # return the data section reshaped to the sizes
    # every failed check -> raise ValueError
    if len(data) < 4:
        raise ValueError("file too short for an IDX header")
    if data[0] != 0 or data[1] != 0:
        raise ValueError("bad magic number")
    if data[2] != 8:
        raise ValueError(f"not a uint8 IDX file (type code {data[2]:#04x})")
    ndim = data[3]

    if len(data) < 4 + 4 * ndim:
        raise ValueError("file too short for an IDX header")

    dims = np.frombuffer(data, dtype='>u4', count=ndim, offset=4)

    header_len = 4 + 4 * ndim
    expected = header_len + int(np.prod(dims))
    if len(data) != expected:
        raise ValueError(f"size mismatch: header says {expected} bytes, file has {len(data)}")

    return np.frombuffer(data, dtype=np.uint8, offset=header_len).reshape(tuple(dims))

# unzip one cached IDX file, then parse it
def _load(path):
    with gzip.open(path, "rb") as f:
        return _parse_idx(f.read())

def get_fashion_mnist(normalize=True, flatten=True, data_home=None):
    """Same signature, return order, shapes and dtypes as get_mnist."""
    # pick cache dir (data_home or _CACHE_DIR), make a subfolder
    # download any of the 4 files not already cached
    # gunzip + parse each, convert X to float32 and y to int32
    # normalize / reshape exactly the way get_mnist does
    cache_dir = data_home if data_home is not None else _CACHE_DIR
    cache_dir = os.path.join(cache_dir, "fashion_mnist")
    os.makedirs(cache_dir, exist_ok=True)

    arrays = {}
    for key, filename in _FILES.items():
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            _download(_BASE_URL + filename, path)
        arrays[key] = _load(path)

    X_train = arrays["train_images"].astype(np.float32)
    X_test = arrays["test_images"].astype(np.float32)
    y_train = arrays["train_labels"].astype(np.int32)
    y_test = arrays["test_labels"].astype(np.int32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if flatten:
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
