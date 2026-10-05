"""Fashion-MNIST loader. Works the same way as get_mnist()."""

import gzip
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

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


def download(name, folder):
    """Download one file, unless it is already in the cache folder."""
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path

    print("Downloading", name)
    temp_path = path + ".part"
    try:
        # Save to a temporary file first
        urllib.request.urlretrieve(URL + name, temp_path)
        # Rename only when the download is complete
        os.replace(temp_path, path)
    except BaseException:
        # Download failed or was stopped: delete the unfinished file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return path


def read_idx(data):
    """Turn the bytes of an IDX file into a NumPy array.

    IDX header:
      byte 0 and 1 : always 0
      byte 2       : data type (8 means uint8)
      byte 3       : number of dimensions
      then 4 bytes for the size of each dimension
    After the header come the pixel or label values.
    """
    if len(data) < 4 or data[0] != 0 or data[1] != 0:
        raise ValueError("This is not an IDX file")
    if data[2] != 8:
        raise ValueError("IDX file is not uint8")

    ndim = data[3]
    header_size = 4 + 4 * ndim
    if ndim == 0 or len(data) < header_size:
        raise ValueError("IDX header is broken")

    # Read each dimension size (big-endian 4-byte number)
    shape = []
    for i in range(ndim):
        start = 4 + 4 * i
        shape.append(int.from_bytes(data[start:start + 4], "big"))

    # The file size must match the shape written in the header
    if len(data) - header_size != int(np.prod(shape)):
        raise ValueError("IDX file size does not match its header")

    values = np.frombuffer(data, dtype=np.uint8, offset=header_size)
    return values.reshape(shape)


def load_file(path):
    """Unzip a .gz file and read it as IDX."""
    with gzip.open(path, "rb") as f:
        return read_idx(f.read())


def get_fashion_mnist(normalize=True, flatten=True, data_home=None):
    """Load Fashion-MNIST.

    Returns (X_train, X_test, y_train, y_test), same as get_mnist():
    images are float32, labels are int32,
    images are (N, 784) if flatten=True, else (N, 1, 28, 28).
    """
    if data_home is None:
        data_home = os.path.join(os.path.expanduser("~"), ".cache", "dptiny")
    folder = os.path.join(data_home, "fashion_mnist")
    os.makedirs(folder, exist_ok=True)

    train_images = load_file(download(FILES[0], folder))
    train_labels = load_file(download(FILES[1], folder))
    test_images = load_file(download(FILES[2], folder))
    test_labels = load_file(download(FILES[3], folder))

    X_train = train_images.reshape(-1, 784).astype(np.float32)
    X_test = test_images.reshape(-1, 784).astype(np.float32)
    y_train = train_labels.astype(np.int32)
    y_test = test_labels.astype(np.int32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    if not flatten:
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)

    return X_train, X_test, y_train, y_test
