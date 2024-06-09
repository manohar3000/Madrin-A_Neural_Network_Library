import numpy as np
import sys
class Linear():
    """
    A fully connected linear layer.

    Attributes:
    no_of_neurons (int): The number of neurons in the layer.
    input_size (int): The size of each input sample.
    weights (np.ndarray): The weight matrix of the layer.
    biases (np.ndarray): The bias vector of the layer (optional).
    gradient_w (np.ndarray): The gradient of the weights.
    gradient_b (np.ndarray): The gradient of the biases.
    """
    
    def __init__(self, no_of_neurons, input_size, gain=1.0, bias=True):
        """
        Initializes the Linear layer.

        Args:
        no_of_neurons (int): Number of neurons in the layer.
        input_size (int): Number of input features.
        gain (float): Scaling factor for weight initialization (default: 1.0).
        bias (bool): Whether to include a bias term (default: True).
        """

        self.no_of_neurons=no_of_neurons
        self.input_size=input_size
        
        # Initialize weights with a random normal distribution scaled by the gain and input size.
        self.weights=np.random.randn(input_size,no_of_neurons) * (gain/(input_size)**0.5)
        self.biases=np.zeros((1,no_of_neurons)) if bias else None

        # Initialize gradients for weights and biases.
        self.gradient_w=np.zeros((input_size,no_of_neurons))
        self.gradient_b=np.zeros((1,no_of_neurons)) if bias else None

    def forward(self,inputs):
        """
        Performs the forward pass through the layer.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the layer.
        """
        self.inputs = inputs
        out = np.dot(inputs,self.weights)
        if self.biases is not None:
            out+=self.biases
        return out

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the layer.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate for gradient descent.

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        self.gradient_w = (1/len(gradient_outputs))*np.dot(self.inputs.T,gradient_outputs)
        self.gradient_b = np.mean(gradient_outputs, axis=0, keepdims= True) if self.biases is not None else None
        self.gradient_descent(lr)
        return np.dot(gradient_outputs,self.weights.T)

    def gradient_descent(self,lr):
        """
        Updates the layer's parameters using gradient descent.

        Args:
        lr (float): The learning rate.
        """
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j]-=lr*self.gradient_w[i][j]
        if self.biases is not None:
            self.biases-=lr*self.gradient_b

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: A list containing the weights and biases (if any).
        """
        return [self.weights] + ([] if self.biases is None else [self.biases])


class Sigmoid():
    """A sigmoid activation layer."""
    
    def forward(self,inputs):
        """
        Performs the forward pass through the sigmoid activation function.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the sigmoid function.
        """
        self.activations = np.exp(np.fmin(inputs, 0)) / (1 + np.exp(-np.abs(inputs)))
        return self.activations

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the sigmoid activation function.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate (not used in this function).

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        A_prime = self.activations*(1-self.activations)
        return A_prime * gradient_outputs

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: An empty list (sigmoid activation has no parameters).
        """
        return []

class Relu():
    """
    A Rectified Linear Unit (ReLU) activation layer.
    """
    def forward(self,inputs):
        """
        Performs the forward pass through the ReLU activation function.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the ReLU function.
        """
        self.activations = np.maximum(np.zeros(inputs.shape),inputs)
        return self.activations

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the ReLU activation function.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate (not used in this function).

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        return gradient_outputs * (self.activations > 0)

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: An empty list (ReLU activation has no parameters).
        """
        return []

class Tanh():
    """
    A hyperbolic tangent (tanh) activation layer.
    """
    def forward(self,inputs):
        """
        Performs the forward pass through the tanh activation function.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the tanh function.
        """
        self.activations = np.tanh(inputs)
        return self.activations

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the tanh activation function.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate (not used in this function).

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        return gradient_outputs * (1 - self.activations ** 2)

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: An empty list (tanh activation has no parameters).
        """
        return []

class Softmax():
    """
    A softmax activation layer.
    """
    def forward(self, inputs):
        """
        Performs the forward pass through the softmax activation function.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the softmax function.
        """
        expo = np.exp(inputs -  np.max(inputs, axis=1, keepdims=True))
        self.activations = expo / np.sum(expo, axis=1, keepdims=True)
        return self.activations

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the softmax activation function.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate (not used in this function).

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        num_samples = self.activations.shape[0]
        num_classes = self.activations.shape[1]

        # Initialize the gradient with respect to inputs
        gradient_inputs = np.zeros_like(self.activations)

        for i in range(num_samples):
            # Flatten activations and gradient for easier manipulation
            s = self.activations[i].reshape(-1, 1)
            d_out = gradient_outputs[i].reshape(-1, 1)

            # Compute the Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)

            # Compute the gradient with respect to the input
            gradient_inputs[i] = np.dot(jacobian_matrix, d_out).reshape(-1)

        return gradient_inputs

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: An empty list (softmax activation has no parameters).
        """
        return []


class LeakyRelu():
    """
    A Leaky Rectified Linear Unit (Leaky ReLU) activation layer.
    """
    def __init__(self, alpha=0.01):
        """
        Initializes the LeakyReLU layer.

        Args:
        alpha (float): Slope of the function for x < 0 (default: 0.01).
        """
        self.alpha = alpha
        self.inputs = None

    def forward(self, inputs):
        """
        Performs the forward pass through the Leaky ReLU activation function.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the Leaky ReLU function.
        """
        self.inputs = inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self,gradient_outputs,lr):
        """
        Performs the backward pass through the Leaky ReLU activation function.

        Args:
        gradient_outputs (np.ndarray): The gradient of the loss with respect to the layer's output.
        lr (float): The learning rate (not used in this function).

        Returns:
        np.ndarray: The gradient of the loss with respect to the layer's input.
        """
        dx = np.ones_like(self.inputs)
        dx[self.inputs < 0] = self.alpha
        return dx * gradient_outputs

    def parameters(self):
        """
        Returns the parameters of the layer.

        Returns:
        list: An empty list (Leaky ReLU activation has no parameters).
        """
        return []


class Network():
    """
    A neural network composed of multiple layers.

    Attributes:
    layers (list): A list of layer instances that form the network.
    costs (list): A list to store the cost at each epoch if tracking is enabled.
    """
    
    def __init__(self, layers):
        """
        Initializes the Network with the given layers.

        Args:
        layers (list): List of layers to include in the network.
        """
        self.layers = layers
        self.costs=[]

    def forward(self,inputs):
        """
        Performs the forward pass through the network.

        Args:
        inputs (np.ndarray): The input data.

        Returns:
        np.ndarray: The output of the network.
        """
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs

    def compile(self, loss, lr):
        """
        Compiles the network with a loss function and learning rate.

        Args:
        loss (str): The loss function to use ('mse' or 'categorical_crossentropy').
        lr (float): The learning rate.
        """
        self.loss=loss
        self.lr=lr

    def backward(self,x,y,lr):
        """
        Performs the backward pass through the network.

        Args:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        lr (float): The learning rate.
        """
        gradient_outputs=self.derivative_of_loss(x,y)
        for layer in reversed(self.layers):
            gradient_outputs=layer.backward(gradient_outputs,lr)

    def derivative_of_loss(self,samples,labels):
        """
        Computes the derivative of the loss function.

        Args:
        samples (np.ndarray): The input data.
        labels (np.ndarray): The target data.

        Returns:
        np.ndarray: The gradient of the loss with respect to the network's output.
        """
        outputs = self.forward(samples)
        if self.loss == 'mse':
            return 2*(outputs-labels)

        elif self.loss == 'categorical_crossentropy':
            one_hot=np.zeros((labels.shape[0], self.layers[-1].activations.shape[1]))
            one_hot[np.arange(labels.shape[0]), labels] = 1
            clipped_outputs = np.clip(outputs, 1e-15, 1 - 1e-15)
            return -(one_hot/clipped_outputs)

        else:
            sys.exit("Specify a valid loss function(mse or categorical_crossentropy)")

    def fit(self, x, y, epochs,batch_size=None,track_loss=False):
        """
        Trains the network on the given data.

        Args:
        x (np.ndarray): The input data.
        y (np.ndarray): The target data.
        epochs (int): The number of epochs to train for.
        batch_size (int): The size of each mini-batch (default: None for full-batch).
        track_loss (bool): Whether to track the loss at each epoch (default: False).
        """
        if batch_size is None:
            for i in range(epochs):
                self.backward(x,y,self.lr)
                if track_loss:
                    self.costs.append(self.cost(x,y))

        else:
            for i in range(epochs):
                for j in range(0,len(x),batch_size):
                    x_mini = x[j:j+batch_size]
                    y_mini = y[j:j+batch_size]
                    self.backward(x_mini,y_mini,self.lr)
                if track_loss:
                    self.costs.append(self.cost(x,y))

    def cost(self,samples,labels):
        """
        Computes the cost (loss) of the network on the given data.

        Args:
        samples (np.ndarray): The input data.
        labels (np.ndarray): The target data.

        Returns:
        float: The computed cost.
        """
        outputs = self.forward(samples)
        if self.loss == 'mse':
            return np.mean(np.sum(((outputs-labels)**2),axis=1,keepdims=True))

        elif self.loss == 'categorical_crossentropy':
            clipped_outputs = np.clip(outputs, 1e-15, 1 - 1e-15)
            correct_class_probs = clipped_outputs[np.arange(samples.shape[0]), labels]
            return np.mean(-np.log(correct_class_probs))

        else:
            print("Specify a valid loss function(mse or categorical_crossentropy)")

    def parameters(self):
        """
        Returns the parameters of all layers in the network.

        Returns:
        list: A list containing the parameters of each layer.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def n_parameters(self):
        """
        Returns the total number of parameters in the network.

        Returns:
        int: The total number of parameters.
        """
        parameters = self.parameters()
        total_parameters=0
        for p in parameters:
            total_parameters+=p.size
        return total_parameters
