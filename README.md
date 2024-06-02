# Madrin
![Logo](images/logo.png)<br>
A cute Neural Network library with Keras-like API with just 100 lines of code. Build for fun and educational purposes. Because the code is so simple, it is very easy to change to your needs. Still under active development. 

## Dependencies
- [numpy](https://numpy.org/install/) 

## Installation
```shell 
pip install madrin
```

## Demo

**Create a neural network:**<br>
You can create a Neural Network by passing a list of layers to the `Network` constructor.<br>
Currently it supports the following layers:
>`Linear(no_of_neurons, input_size)`<br>
>`Relu()`<br>
>`Sigmoid()`<br>
>`Tanh()`<br>
>`Softmax()`<br>

```python
import numpy as np

# Import the necessary classes from the madrin library
from madrin import Linear, Sigmoid, Relu, Tanh, Softmax, Network

# Generate some dummy data for training
np.random.seed(0)  # For reproducibility
X_train = np.random.randn(100, 3)  # 100 samples, 3 features each
y_train = np.random.randint(0, 3, 100)  # 100 labels (3 classes)

# Create the network
nn = Network([
    Linear(no_of_neurons=5, input_size=3),  # First layer: 3 input features, 5 neurons
    Relu(),  # ReLU activation
    Linear(no_of_neurons=3, input_size=5),  # Second layer: 5 input features, 3 neurons (output layer)
    Softmax()  # Softmax activation for multi-class classification
])

# Compile the network with loss function and learning rate
nn.compile(loss='categorical_crossentropy', lr=0.01)

# Train the network
nn.fit(X_train, y_train, epochs=1000)

# Make predictions
predictions = nn.forward(X_train)

# Print the predictions
print(predictions)

# Print the training costs
import matplotlib.pyplot as plt
plt.plot(nn.costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost Over Time')
plt.show()

```
### Contributing
Contributions are welcome! Please open an issue or submit a pull request on Github.

### License
Madrin is released under the MIT License.
