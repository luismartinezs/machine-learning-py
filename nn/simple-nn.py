import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


input_layer_neurons = 4  # Number of features in input data
hidden_layer_neurons = 3  # Number of neurons in the hidden layer
output_neurons = 1  # Number of neurons in the output layer

# Weight matrices
weights_input_hidden = np.random.uniform(
    size=(input_layer_neurons, hidden_layer_neurons)
)
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Biases
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))


def forward_propagation(inputs):
    # Input to Hidden Layer
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)

    # Hidden Layer to Output Layer
    output_layer_input = (
        np.dot(hidden_layer_activation, weights_hidden_output) + bias_output
    )
    output = sigmoid(output_layer_input)

    return hidden_layer_activation, output


def backward_propagation(inputs, hidden_layer_activation, output, expected_output):
    global weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

    # Calculate the error
    error = expected_output - output
    d_output = error * sigmoid_derivative(output)

    # Calculate the error for the hidden layer
    hidden_error = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = hidden_error * sigmoid_derivative(hidden_layer_activation)

    # Update the weights and biases
    weights_hidden_output += hidden_layer_activation.T.dot(d_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate

    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


def train(inputs, expected_output, epochs):
    for epoch in range(epochs):
        hidden_layer_activation, output = forward_propagation(inputs)
        backward_propagation(inputs, hidden_layer_activation, output, expected_output)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Error: {np.mean(np.abs(expected_output - output))}")


def normalize(data):
    """Normalize the data to a range between 0 and 1."""
    max_values = np.max(data, axis=0)
    normalized_data = data / max_values
    return normalized_data, max_values


def denormalize(normalized_data, max_values):
    """Denormalize the data from a range between 0 and 1 to the original scale."""
    denormalized_data = normalized_data * max_values
    return denormalized_data


# Example data (normalized)
# Features: [area, bedrooms, bathrooms, age]
inputs = np.array(
    [
        [2104, 3, 1, 45],
        [1600, 3, 2, 40],
        [2400, 3, 3, 30],
        [1416, 2, 2, 20],
        [3000, 4, 3, 15],
    ]
)

# Example expected output (normalized)
expected_output = np.array([[399900], [329900], [369000], [232000], [539900]])

# Normalize inputs and outputs
inputs, max_input_values = normalize(inputs)
expected_output, max_output_values = normalize(expected_output)

learning_rate = 0.1
epochs = 10000

train(inputs, expected_output, epochs)

# Test the network
_, predicted_output = forward_propagation(inputs)
denormalized_output = denormalize(predicted_output, max_output_values)
print("Denormalized Predicted Output:\n", denormalized_output)
