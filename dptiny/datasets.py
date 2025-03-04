import numpy as np
from sklearn.datasets import fetch_openml

def get_mnist(normalize=True, flatten=True):
    """Get MNIST dataset.
    
    Args:
        normalize (bool): If True, normalize pixels to [0, 1].
        flatten (bool): If True, flatten images to vectors.
    
    Returns:
        tuple: Training data, test data, training labels, test labels.
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)

    if normalize:
        X = X / 255.0

    if not flatten:
        X = X.reshape(-1, 1, 28, 28)

    train_size = 60000
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.data_size = len(dataset[0])
        self.max_iter = self.data_size // batch_size
        if not self.drop_last and self.data_size % batch_size != 0:
            self.max_iter += 1
        self.reset()
        
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)
            
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:min((i + 1) * batch_size, self.data_size)]
        batch_x = self.dataset[0][batch_index]
        batch_t = self.dataset[1][batch_index]
        self.iteration += 1
        
        return batch_x, batch_t
