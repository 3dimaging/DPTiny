# DPTiny

A minimal implementation of a deep learning framework for educational purposes.

## Features

- Automatic differentiation (autograd) system
- Basic mathematical operations with operator overloading
- NumPy-based computation

## Installation

```bash
pip install -e .
```

## Usage Example

```python
import numpy as np
from dptiny import Variable

# Create variables
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

# Perform operations
z = x * y
print(z.data)  # 6.0

# Compute gradients
z.backward()
print(x.grad)  # 3.0
print(y.grad)  # 2.0
```

## Architecture

```mermaid
classDiagram
    class Config {
        +bool enable_backprop
    }

    class Variable {
        -np.ndarray _grad
        +np.ndarray data
        +str name
        +int generation
        +Function creator
        +tuple shape
        +int ndim
        +int size
        +grad
        +cleargrad()
        +backward(retain_grad)
        +__matmul__(other)
    }

    class Function {
        <<abstract>>
        +int generation
        +list inputs
        +list outputs
        +__call__(*inputs)
        +forward(*xs)*
        +backward(*gys)*
    }

    class Layer {
        <<abstract>>
        +dict params
        +dict grads
        +__call__(*inputs)
        +forward(inputs)*
        +backward(grads)*
        +cleargrads()
    }

    class Add {
        +forward(x0, x1)
        +backward(gy)
    }

    class Mul {
        +forward(x0, x1)
        +backward(gy)
    }

    class MatMul {
        +forward(x, W)
        +backward(gy)
    }

    class Exp {
        +forward(x)
        +backward(gy)
    }

    class Log {
        +forward(x)
        +backward(gy)
    }

    class Sigmoid {
        +forward(x)
        +backward(gy)
    }

    class ReLU {
        +forward(x)
        +backward(gy)
    }

    class Softmax {
        +forward(x)
        +backward(gy)
    }

    class SoftmaxCrossEntropy {
        +forward(x, t)
        +backward(gy)
    }

    class Neg {
        +forward(x)
        +backward(gy)
    }

    class Sub {
        +forward(x0, x1)
        +backward(gy)
    }

    class Div {
        +forward(x0, x1)
        +backward(gy)
    }

    class Linear {
        +int in_size
        +int out_size
        +Variable W
        +Variable b
        +forward(x)
        +backward(grads)
    }

    class MLP {
        +list layers
        +forward(x)
        +backward(grads)
        +predict(x)
        +accuracy(x, t)
    }

    class SGD {
        +float lr
        +float momentum
        +dict vs
        +update(params)
        +set_lr(lr)
    }

    class DataLoader {
        +tuple dataset
        +int batch_size
        +bool shuffle
        +int max_iter
        +int iteration
        +__iter__()
        +__next__()
        +reset()
    }

    Variable --> Function : creator
    Function --> Variable : inputs/outputs
    Function <|-- Add
    Function <|-- Mul
    Function <|-- MatMul
    Function <|-- Exp
    Function <|-- Log
    Function <|-- Sigmoid
    Function <|-- ReLU
    Function <|-- Softmax
    Function <|-- SoftmaxCrossEntropy
    Function <|-- Neg
    Function <|-- Sub
    Function <|-- Div
    Layer <|-- Linear
    Layer <|-- ReLU
    Layer <|-- MLP
    Layer --> Variable : params
    MLP --> Linear : contains
    MLP --> ReLU : contains
```

## Requirements

- NumPy
