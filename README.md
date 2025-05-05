# 🧠 Madrin

<div align="center">

![Logo](images/logo.png)

**Elegant Neural Networks, Beautifully Simple**

[![PyPI version](https://img.shields.io/badge/pip-install%20madrin-blue)](https://pypi.org/project/madrin/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()

</div>

## ✨ Overview

Madrin is a lightweight neural network library designed with clarity and simplicity at its core. With an intuitive Keras-inspired API, it serves as both a practical tool and an educational platform for those looking to understand the inner workings of neural networks.

What makes Madrin special:
- **Transparent Implementation**: Clean code that's easy to read, understand, and modify
- **Educational Focus**: Perfect for learning neural network fundamentals
- **Flexibility**: Simple enough to customize for your specific requirements
- **Minimalist Dependencies**: Built primarily on NumPy for maximum compatibility

*Madrin makes neural networks approachable without sacrificing functionality.*

## 📦 Installation

```bash
pip install madrin
```

That's it! Just one command and you're ready to start building neural networks.

## 🛠️ Dependencies

- [NumPy](https://numpy.org/install/) - The only external dependency you'll need

## 🚀 Quick Start

Build your first neural network in just a few lines of code:

```python
import numpy as np
from madrin import Network, Linear, Relu, Softmax

# Create a simple classification network
model = Network([
    Linear(no_of_neurons=64, input_size=10),  # Hidden layer
    Relu(),                                   # Activation
    Linear(no_of_neurons=3, input_size=64),   # Output layer
    Softmax()                                 # Output activation
])

# Compile with your preferred settings
model.compile(loss='categorical_crossentropy', lr=0.01)

# Train and predict!
model.fit(X_train, y_train, epochs=100)
predictions = model.forward(X_test)
```

## 🧩 Available Layers

Madrin provides a growing collection of essential neural network components:

| Layer | Description |
|-------|-------------|
| `Linear(neurons, input_size)` | Fully connected layer |
| `Relu()` | Rectified Linear Unit activation |
| `LeakyRelu()` | Leaky ReLU for preventing "dying ReLU" |
| `Sigmoid()` | Sigmoid activation for binary classification |
| `Tanh()` | Hyperbolic tangent activation |
| `Softmax()` | Softmax activation for multi-class problems |

## 📊 Complete Example

Here's how to create, train, and evaluate a complete neural network:

```python
import numpy as np
import matplotlib.pyplot as plt
from madrin import Linear, Relu, Softmax, Network

# Prepare data
np.random.seed(0)  # For reproducibility
X_train = np.random.randn(1000, 3)  # 1000 samples, 3 features each
y_train = np.random.randint(0, 3, 1000)  # 1000 labels (3 classes)

# Build network architecture
model = Network([
    Linear(no_of_neurons=5, input_size=3),
    Relu(),
    Linear(no_of_neurons=3, input_size=5),
    Softmax()
])

# Check model complexity
print(f"Model has {model.n_parameters()} trainable parameters")

# Compile and train
model.compile(loss='categorical_crossentropy', lr=0.01)
model.fit(X_train, y_train, epochs=1000, batch_size=100, track_loss=True)

# Make predictions
predictions = model.forward(X_train)
print("Prediction shape:", predictions.shape)

# Visualize training progress
plt.figure(figsize=(10, 6))
plt.plot(model.costs)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14)
plt.grid(alpha=0.3)
plt.show()
```

## 👥 Contributing

Contributions are warmly welcomed! Madrin is designed to grow with community input. Whether you're fixing bugs, adding features, or improving documentation:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

For major changes or ideas, please open an issue first to discuss what you'd like to change.

Visit our [GitHub repository](https://github.com/manohar3000/Madrin-A_Neural_Network_Library) to get started.

## 📄 License

Madrin is released under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  
**Built with ❤️ for simplicity and understanding**

</div>
