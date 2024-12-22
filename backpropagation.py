Б1. Backpropagation код нейронної мережі
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) Activation functions
# ============================================================
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
	# derivative of sigmoid if x = sigmoid(u)
	return x * (1.0 - x)

# ============================================================
# 2) Neural Network class
# ============================================================
class NeuralNetwork:
	"""
	A simple feedforward neural network with backpropagation:
	- Single hidden layer
	- Sigmoid activation
	- Gradient descent
	"""

	def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
    	"""
    	input_dim  : number of input neurons
    	hidden_dim : number of neurons in the hidden layer
    	output_dim : number of output neurons
    	lr     	: learning rate
    	"""
    	self.lr = lr
   	 
    	# Random initialization of weights in [-1,1], or more advanced methods (Xavier, He, etc.)
    	self.W1 = np.random.uniform(-1.0, 1.0, (input_dim, hidden_dim))
    	self.b1 = np.zeros((1, hidden_dim))
   	 
    	self.W2 = np.random.uniform(-1.0, 1.0, (hidden_dim, output_dim))
    	self.b2 = np.zeros((1, output_dim))

	def forward(self, X):
    	"""
    	Forward pass:
    	X   : input matrix (N, input_dim)
    	Z1  : hidden layer activation
    	Z2  : final output
    	"""
    	# Hidden layer
    	self.Z1_lin = np.dot(X, self.W1) + self.b1
    	self.Z1 = sigmoid(self.Z1_lin)
   	 
    	# Output layer
    	self.Z2_lin = np.dot(self.Z1, self.W2) + self.b2
    	self.Z2 = sigmoid(self.Z2_lin)
   	 
    	return self.Z2

	def backward(self, X, y, output):
    	"""
    	Backpropagation step to update W1, W2 and b1, b2.
   	 
    	X  	: (N, input_dim)
    	y  	: (N, output_dim) — ground truth
    	output : (N, output_dim) — forward pass result
    	"""
    	N = X.shape[0]
   	 
    	# Output layer error
    	error_output = output - y
    	dZ2 = error_output * sigmoid_derivative(output)  # element-wise
   	 
    	# Grad for W2, b2
    	dW2 = np.dot(self.Z1.T, dZ2) / N
    	db2 = np.sum(dZ2, axis=0, keepdims=True) / N
   	 
    	# Hidden layer error
    	error_hidden = np.dot(dZ2, self.W2.T)
    	dZ1 = error_hidden * sigmoid_derivative(self.Z1)
   	 
    	# Grad for W1, b1
    	dW1 = np.dot(X.T, dZ1) / N
    	db1 = np.sum(dZ1, axis=0, keepdims=True) / N
   	 
    	# Update weights
    	self.W2 -= self.lr * dW2
    	self.b2 -= self.lr * db2
    	self.W1 -= self.lr * dW1
    	self.b1 -= self.lr * db1

	def train(self, X, y, epochs=10000):
    	"""
    	Training loop: forward + backward pass in a loop.
    	"""
    	for epoch in range(epochs):
        	# forward
        	output = self.forward(X)
        	# backward
        	self.backward(X, y, output)
       	 
        	# optional: track loss
        	if (epoch+1) % 1000 == 0:
            	loss = np.mean((output - y)**2)
            	print(f"Epoch {epoch+1}, Loss={loss:.5f}")

	def predict(self, X):
    	"""
    	Use the trained network to predict labels (binary classification).
    	"""
    	output = self.forward(X)
    	return (output > 0.5).astype(int)

# ============================================================
# 3) Minimal "drawing" of the network architecture
# ============================================================
def draw_network(input_dim, hidden_dim, output_dim):
	"""
	Draws a minimalistic diagram (input -> hidden -> output).
	Each "neuron" is a circle, edges are lines.
	"""

	plt.figure(figsize=(6,4))
	ax = plt.gca()
	ax.set_xlim([0, 8])
	ax.set_ylim([0, 6])
	ax.set_aspect('equal')
	plt.axis('off')

	# Coordinates for each layer
	# input layer (x=1), hidden layer (x=4), output layer (x=7)
	x_input = 1
	x_hidden = 4
	x_output = 7

	# We'll space neurons vertically
	# For convenience, center them around y=3
	y_input = np.linspace(3 - (input_dim-1), 3 + (input_dim-1), input_dim)
	y_hidden = np.linspace(3 - (hidden_dim-1), 3 + (hidden_dim-1), hidden_dim)
	y_output = np.linspace(3 - (output_dim-1), 3 + (output_dim-1), output_dim)

	# Draw input neurons
	for i in range(input_dim):
    	circle = plt.Circle((x_input, y_input[i]), 0.2, fill=True, color='skyblue')
    	ax.add_patch(circle)
    	ax.text(x_input - 0.4, y_input[i], f"In {i+1}", va='center', fontsize=9)

	# Draw hidden neurons
	for h in range(hidden_dim):
    	circle = plt.Circle((x_hidden, y_hidden[h]), 0.2, fill=True, color='lightgreen')
    	ax.add_patch(circle)
    	ax.text(x_hidden + 0.3, y_hidden[h], f"H {h+1}", va='center', fontsize=9)

	# Draw output neurons
	for j in range(output_dim):
    	circle = plt.Circle((x_output, y_output[j]), 0.2, fill=True, color='salmon')
    	ax.add_patch(circle)
    	ax.text(x_output + 0.3, y_output[j], f"Out {j+1}", va='center', fontsize=9)

	# Draw connections from input to hidden
	for i in range(input_dim):
    	for h in range(hidden_dim):
        	ax.plot([x_input + 0.2, x_hidden - 0.2],
                	[y_input[i], y_hidden[h]], 'k-', linewidth=1, alpha=0.6)

	# Draw connections from hidden to output
	for h in range(hidden_dim):
    	for j in range(output_dim):
        	ax.plot([x_hidden + 0.2, x_output - 0.2],
                	[y_hidden[h], y_output[j]], 'k-', linewidth=1, alpha=0.6)

	plt.title("Minimal Backprop NN Diagram\n(input -> hidden -> output)")
	plt.show()

# ============================================================
# 4) Usage Example: XOR
# ============================================================
if __name__ == '__main__':

	# 4.1 Draw the network architecture
	draw_network(input_dim=2, hidden_dim=2, output_dim=1)

	# 4.2 XOR data
	X = np.array([
    	[0,0],
    	[0,1],
    	[1,0],
    	[1,1]
	])
	# Targets (XOR => 0,1,1,0)
	y = np.array([[0],[1],[1],[0]])
    
	# 4.3 Create and train the network
	nn = NeuralNetwork(input_dim=2, hidden_dim=2, output_dim=1, lr=0.5)
	nn.train(X, y, epochs=5000)
    
	# 4.4 Test predictions
	predictions = nn.predict(X)
	print("\nXOR predictions:")
	for i in range(len(X)):
    	print(f"Input: {X[i]}, Predicted: {predictions[i][0]}, True: {y[i][0]}")
