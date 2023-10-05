import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration
np.random.seed(42)
num_samples = 100
X = np.random.rand(num_samples, 2) * 10 - 5
Y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# Initialize weights and bias
w = np.random.rand(2)
b = np.random.rand()

# Perceptron learning rate
learning_rate = 0.1

# Number of training iterations
num_iterations = 10

# Create a figure for visualizations
fig, ax = plt.subplots(figsize=(8, 8))

# Training loop
for iteration in range(num_iterations):
    misclassified = 0

    # Iterate through each data point
    for i in range(num_samples):
        x = X[i]
        y = Y[i]

        # Compute the perceptron output
        z = np.dot(w, x) + b
        if z >= 0:
            predicted = 1
        else:
            predicted = -1

        # Update weights and bias if misclassified
        if predicted != y:
            w += learning_rate * y * x
            b += learning_rate * y
            misclassified += 1

        # Visualize the data points and the perceptron line
        ax.clear()
        ax.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], label='Class 1', marker='o')
        ax.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], label='Class -1', marker='x')
        ax.set_title(f'Iteration {iteration + 1}')
        
        # Plot the perceptron line
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        x_vals = np.linspace(-5, 5, 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, '-r', label='Perceptron Line')
        
        ax.legend()
        plt.pause(0.1)

        plt.savefig(f'loop_{iteration+1}')

    # If no misclassified points, stop training
    if misclassified == 0:
        break

plt.show()
