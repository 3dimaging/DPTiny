# DPTiny: A Minimal Deep Learning Framework
## Slide Deck for Code Walkthrough

---

## Slide 1: Introduction

### What is DPTiny?
- A **minimal** deep learning framework for **educational purposes**
- Implements automatic differentiation (autograd) from scratch
- NumPy-based computation (no GPU support)
- ~500 lines of Python code

### Why study this code?
- Understand how PyTorch/TensorFlow work under the hood
- Learn automatic differentiation principles
- See clean OOP design in ML

---

## Slide 2: Project Structure

```
DPTiny/
├── dptiny/
│   ├── __init__.py       # Public API exports
│   ├── core.py           # Variable, Config (the heart)
│   ├── functions.py      # Mathematical operations
│   ├── layers.py         # Neural network layers
│   └── datasets.py       # Data loading utilities
├── examples/
│   ├── matmul_example.py # Matrix multiplication demo
│   └── mnist.py          # Full training example
└── README.md
```

**Key insight**: Each file has a single responsibility

---

## Slide 3: The Variable Class - Data with Memory

Location: `dptiny/core.py` (lines 17-93)

```python
class Variable:
    def __init__(self, data, name=None):
        self.data = data          # NumPy array
        self.name = name          # For debugging
        self._grad = None         # Gradient (initially None)
        self.creator = None       # Function that created this variable
        self.generation = 0       # For topological ordering
```

### Key Properties
- `grad` - stores computed gradients (lazy initialization)
- `creator` - reference to the `Function` that produced this variable
- `generation` - helps order the backward pass

### Think of it as:
> A wrapper around a NumPy array that remembers how it was created

---

## Slide 4: The Computational Graph

### How Variables Connect

```
Input Variable    Function      Output Variable
     x    ───▶    Add()    ───▶      y
                  ▲  ▲
                  │  │
             x0 ─┘  └── x1
```

**Key relationship**: `y.creator = Add()`

### Tracing the Graph
```python
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = x * y  # z.creator = Mul()

# The graph: z → Mul → [x, y]
```

---

## Slide 5: The Function Base Class

Location: `dptiny/functions.py` (lines 6-31)

```python
class Function(ABC):
    def __call__(self, *inputs):
        # 1. Convert inputs to Variables
        inputs = [as_variable(x) for x in inputs]
        
        # 2. Forward pass (actual computation)
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        
        # 3. Create output Variables
        outputs = [Variable(as_array(y)) for y in ys]
        
        # 4. Connect to graph (if training)
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.creator = self  # Link back to this function
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
    
    @abstractmethod
    def forward(self, *xs): raise NotImplementedError()
    
    @abstractmethod
    def backward(self, *gys): raise NotImplementedError()
```

---

## Slide 6: Implementing a Function - MatMul

Location: `dptiny/functions.py` (lines 72-89)

```python
class MatMul(Function):
    def forward(self, x, W):
        # Store shapes for backward pass
        self.x_shape = x.shape
        self.W_shape = W.shape
        return x @ W  # NumPy matrix multiplication

    def backward(self, gy):
        # Chain rule: dL/dx = dL/dy @ W.T
        #             dL/dW = x.T @ dL/dy
        x, W = self.inputs
        gx = gy @ W.data.T
        gW = x.data.T @ gy
        return gx, gW
```

### Pattern for Any Function:
1. **Forward**: Compute output, save what's needed for backward
2. **Backward**: Apply chain rule, return gradients w.r.t. inputs

---

## Slide 7: Backpropagation Algorithm

Location: `dptiny/core.py` (lines 56-89)

```python
def backward(self, retain_grad=False):
    # Initialize gradient to ones (dL/dL = 1)
    if self.grad is None:
        self.grad = np.ones_like(self.data)
    
    funcs = []  # Priority queue of functions to process
    seen_set = set()
    
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
            funcs.sort(key=lambda x: x.generation)  # Topological sort!
    
    add_func(self.creator)
    
    while funcs:
        f = funcs.pop()  # Get function with highest generation
        gys = [output().grad for output in f.outputs]  # Get output grads
        gxs = f.backward(*gys)  # Compute input grads via chain rule
        
        for x, gx in zip(f.inputs, gxs):
            if x.grad is None:
                x.grad = gx
            else:
                x.grad = x.grad + gx  # Accumulate gradients
            
            if x.creator is not None:
                add_func(x.creator)  # Continue backprop
```

---

## Slide 8: Generation-Based Ordering

### The Problem
Functions must be processed in **reverse topological order**:
```
x → f → y → g → z → h → w
                ↑
           start here
```

### The Solution: Generation Counter
- Input variables: `generation = 0`
- Each function: `generation = max(input.generations) + 1`

```python
# Example
x = Variable(np.array(2.0))  # gen=0
y = x * 2                   # gen=1 (Mul)
z = y + 3                   # gen=2 (Add)

# Backward pass order: Add → Mul
```

**Key insight**: Higher generation = later in forward pass = earlier in backward pass

---

## Slide 9: Layers - Building Neural Networks

Location: `dptiny/layers.py` (lines 6-88)

```python
class Layer(ABC):
    def __init__(self):
        self.params = {}  # Trainable parameters
        self.grads = {}   # Parameter gradients
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    @abstractmethod
    def forward(self, inputs): pass
    
    @abstractmethod
    def backward(self, grads): pass
    
    def cleargrads(self):
        for param in self.params.values():
            if param is not None:
                param.cleargrad()
```

**Layers are higher-level than Functions**:
- Functions operate on raw data
- Layers hold **state** (parameters like weights, biases)

---

## Slide 10: Linear Layer Implementation

Location: `dptiny/layers.py` (lines 29-53)

```python
class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False):
        super().__init__()
        
        # He initialization (good for ReLU)
        W_data = np.random.randn(in_size, out_size) * np.sqrt(2 / in_size)
        self.W = Variable(W_data.astype(np.float32))
        
        if nobias:
            self.b = None
        else:
            self.b = Variable(np.zeros(out_size, dtype=np.float32))
        
        # Register as parameters
        self.params = {'W': self.W, 'b': self.b}
    
    def forward(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b  # Broadcasting works here
        return y
```

---

## Slide 11: Multi-Layer Perceptron (MLP)

Location: `dptiny/layers.py` (lines 62-99)

```python
class MLP(Layer):
    def __init__(self, in_size, hidden_sizes, out_size):
        super().__init__()
        self.layers = []
        
        sizes = [in_size] + hidden_sizes + [out_size]
        for i in range(len(sizes) - 1):
            # Linear layer
            layer = Linear(sizes[i], sizes[i+1])
            setattr(self, f'l{i+1}', layer)
            self.layers.append(layer)
            
            # Register parameters with prefix
            for key, param in layer.params.items():
                self.params[f'l{i+1}.{key}'] = param
            
            # Activation (except last layer)
            if i != len(sizes) - 2:
                self.layers.append(ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## Slide 12: The Training Loop - MNIST Example

Location: `examples/mnist.py`

```python
# 1. Setup
model = MLP(784, [100], 10)  # Input, hidden, output
optimizer = SGD(lr=0.1)

# 2. Get data
X_train, X_test, y_train, y_test = get_mnist()
train_loader = DataLoader((X_train, y_train), batch_size=100)

# 3. Training loop
for epoch in range(max_epoch):
    for batch_x, batch_t in train_loader:
        # Forward
        y = model(batch_x)
        loss = softmax_cross_entropy(y, batch_t)
        
        # Backward
        model.cleargrads()
        loss.backward()
        
        # Update
        optimizer.update(model.params)
```

---

## Slide 13: Key Takeaways

### Design Patterns Used
1. **Template Method Pattern**: `Function` base class with `forward()`/`backward()`
2. **Composite Pattern**: `MLP` composes multiple `Layer`s
3. **Factory Pattern**: `as_variable()`, `as_array()` conversions

### Core Concepts Learned
- **Computational Graph**: Variables track their creators
- **Reverse Mode AD**: Backward pass uses chain rule
- **Topological Sorting**: Generation counter ensures correct order

### Extensions to Try
- Add `Conv2d` layer
- Implement `BatchNorm`
- Add `Adam` optimizer
- Support GPU (CuPy)

---

## Slide 14: Reading Order for Self-Study

### Beginner Path (2-3 hours)
1. `core.py:17-55` - Variable class
2. `functions.py:6-31` - Function ABC
3. `functions.py:33-62` - Add/Mul (simple functions)
4. `layers.py:6-27` - Layer ABC
5. `examples/matmul_example.py` - Simple usage

### Intermediate Path (+2 hours)
6. `core.py:56-89` - Backprop algorithm
7. `functions.py:72-89` - MatMul (matrix operation)
8. `functions.py:161-188` - SoftmaxCrossEntropy (loss)
9. `layers.py:29-53` - Linear layer
10. `datasets.py:30-64` - DataLoader

### Advanced Path
11. `layers.py:62-99` - MLP composition
12. `examples/mnist.py` - Full training
13. Add your own function/layer!

---

## Questions?

Explore the code at: https://github.com/3dimaging/DPTiny

Happy learning! 🎓
