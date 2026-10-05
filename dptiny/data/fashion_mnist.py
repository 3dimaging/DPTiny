
"""Fashion-MNIST dataset loader."""

import gzip
import os
import urllib.request

import numpy as np


_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "dptiny",
    "fashion_mnist",
)


BASE_URL = (
    "https://raw.githubusercontent.com/"
    "zalandoresearch/fashion-mnist/master/data/fashion/"
)


FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def _download_file(url, destination):
    """Download a file safely, using a temporary file first."""

    # If the file is already downloaded, use the cached copy.
    if os.path.exists(destination):
        return

    # Make sure the cache directory exists.
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Temporary filename used while downloading.
    temp_path = destination + ".part"

    try:
        # Remove an old incomplete download, if one exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Download into the temporary file.
        urllib.request.urlretrieve(url, temp_path)

        # Download succeeded, so replace/rename it to the final filename.
        os.replace(temp_path, destination)

    except Exception:
        # If the download fails, delete the incomplete temporary file.
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise


def _read_idx(path):
    """Read a gzip-compressed uint8 IDX file."""

    with gzip.open(path, "rb") as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError("Invalid IDX file: header is too short.")

    if data[0] != 0 or data[1] != 0 or data[2] != 0x08:
        raise ValueError("Invalid IDX file: expected uint8 IDX format.")

    num_dims = data[3]
    header_size = 4 + 4 * num_dims

    if len(data) < header_size:
        raise ValueError("Invalid IDX file: incomplete header.")

    shape = tuple(
        np.frombuffer(
            data[4:header_size],
            dtype=">u4"
        )
    )

    expected_size = header_size + int(np.prod(shape))

    if len(data) != expected_size:
        raise ValueError("Invalid IDX file: size does not match header.")

    array = np.frombuffer(
        data[header_size:],
        dtype=np.uint8
    )

    return array.reshape(shape)


def get_fashion_mnist(normalize=True, flatten=True, data_home=None):
    """Load the Fashion-MNIST dataset."""

    cache_dir = data_home if data_home is not None else _CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    paths = {}

    for filename in FILES:
        path = os.path.join(cache_dir, filename)
        _download_file(BASE_URL + filename, path)
        paths[filename] = path

    X_train = _read_idx(paths["train-images-idx3-ubyte.gz"])
    y_train = _read_idx(paths["train-labels-idx1-ubyte.gz"])
    X_test = _read_idx(paths["t10k-images-idx3-ubyte.gz"])
    y_test = _read_idx(paths["t10k-labels-idx1-ubyte.gz"])

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
