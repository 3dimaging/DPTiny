import numpy as np
from dptiny.core import Variable, no_grad
from dptiny.functions import relu, matmul
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        return outputs

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, grads):
        raise NotImplementedError()

    def cleargrads(self):
        """Clear gradients of all parameters."""
        for param in self.params.values():
            if param is not None:
                param.cleargrad()

class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        # Initialize weights using He initialization (better for ReLU)
        W_data = np.random.randn(in_size, out_size).astype(np.float32) * np.sqrt(2 / in_size)
        self.W = Variable(W_data)

        if nobias:
            self.b = None
        else:
            self.b = Variable(np.zeros(out_size, dtype=np.float32))  # Shape as (out_size) for better broadcasting
            
        self.params = {'W': self.W, 'b': self.b}
        
    def forward(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b  # Broadcasting will work correctly with shape (out_size)
        return y

    def backward(self, grads):
        raise NotImplementedError()

class ReLU(Layer):
    def forward(self, x):
        return relu(x)

    def backward(self, grads):
        raise NotImplementedError()

class MLP(Layer):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        self.layers = []
        self.params = {}
        
        sizes = [in_size] + hidden_sizes + [out_size]
        for i in range(len(sizes)-1):
            layer = Linear(sizes[i], sizes[i+1])
            setattr(self, f'l{i+1}', layer)
            self.layers.append(layer)
            
            # Add layer parameters to the MLP parameters
            for key, param in layer.params.items():
                self.params[f'l{i+1}.{key}'] = param
            
            if i != len(sizes)-2:  # Not the last layer
                self.layers.append(ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grads):
        raise NotImplementedError()

    def predict(self, x):
        with no_grad():
            y = self.forward(x)
            return y.data.argmax(axis=1)

    def accuracy(self, x, t):
        y = self.predict(x)
        if t.ndim != 1:
            t = t.argmax(axis=1)
        accuracy = (y == t).mean()
        return float(accuracy)


class SGD:
    """Stochastic Gradient Descent optimizer with momentum support."""
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update(self, params):
        """Update parameters using momentum SGD.
        
        Args:
            params (dict): Dictionary of parameters to update
        """
        for name, param in params.items():
            if param is None or param.grad is None:
                continue
                
            # Initialize velocity if not exists
            if name not in self.vs:
                self.vs[name] = np.zeros_like(param.data)
                
            # Update velocity
            self.vs[name] = self.momentum * self.vs[name] - self.lr * param.grad
            
            # Update parameter
            param.data = param.data + self.vs[name]
            
    def set_lr(self, lr):
        """Set learning rate.
        
        Args:
            lr (float): New learning rate
        """
        self.lr = lr
