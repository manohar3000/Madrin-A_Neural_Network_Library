# Madrin
![Logo](images/logo.png)<br>
A cute Neural Network library with Keras-like API. Build for fun and educational purposes. Because the code is so simple, it is very easy to change to your needs. Still under active development.
 
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
>`LeakyRelu()`<br>
>`Sigmoid()`<br>
>`Tanh()`<br>
>`Softmax()`<br>

```python
import numpy as np

# Import the necessary classes from the madrin library
from madrin import Linear, Sigmoid, Relu, LeakyRelu, Tanh, Softmax, Network

# Generate some dummy data for training
np.random.seed(0)  # For reproducibility
X_train = np.random.randn(1000, 3)  # 1000 samples, 3 features each
y_train = np.random.randint(0, 3, 1000)  # 1000 labels (3 classes)

# Create the network
model = Network([
    Linear(no_of_neurons=5, input_size=3),  # First layer: 3 input features, 5 neurons
    Relu(),  # ReLU activation
    Linear(no_of_neurons=3, input_size=5),  # Second layer: 5 input features, 3 neurons (output layer)
    Softmax()  # Softmax activation for multi-class classification
])

# See the total number of trainable parameters(i.e., weights and biases)
print(model.n_parameters())

# Compile the network with loss function and learning rate
model.compile(loss='categorical_crossentropy', lr=0.01)

# Train the network
model.fit(X_train, y_train, epochs=1000, batch_size=100, track_loss = True)

# Make predictions
predictions = model.forward(X_train)

# Print the predictions
print(predictions)

# Print the training costs
import matplotlib.pyplot as plt
plt.plot(np.arange(len(model.costs)),model.costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Training Cost Over Time')
plt.show()

```
### Contributing
Contributions are welcome! Please open an issue or submit a pull request on [Github](https://github.com/manohar3000/Madrin-A_Neural_Network_Library).

### License
Madrin is released under the MIT License.
