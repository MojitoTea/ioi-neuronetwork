import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Define a small adjacency matrix for an example network
#	Suppose we have N cells laid out in a small hex pattern.
# ------------------------------------------------------------

def create_hex_adjacency(rows=4, cols=5):
	"""
	Create adjacency for a small hex grid of 'rows' x 'cols'.
	Returns:
    	A (N,N) adjacency matrix where N = rows * cols.
	"""
	N = rows * cols
	A = np.zeros((N, N), dtype=int)
    
	# For each cell, define who is adjacent in a hex layout:
	# We'll do a simple offset approach for adjacency
	for r in range(rows):
    	for c in range(cols):
        	idx = r * cols + c
        	# Potential neighbors in hex grid
        	neighbors = []
        	# Basic orthonormal neighbors
        	if c+1 < cols:  # right
            	neighbors.append(idx + 1)
        	if c-1 >= 0:  # left
            	neighbors.append(idx - 1)
        	if r+1 < rows:  # below
            	neighbors.append(idx + cols)
        	if r-1 >= 0:  # above
            	neighbors.append(idx - cols)
       	 
        	# Additional diagonal neighbors in hex layout
        	# (depends on row, we shift differently)
        	if r % 2 == 0:  
            	# For even rows, the "upper-right" neighbor is (r-1, c) etc.
            	if r-1 >= 0 and c < cols:
                	neighbors.append(idx - cols)  	# already done above
            	if r-1 >= 0 and (c+1) < cols:
                	neighbors.append(idx - cols + 1)  
            	if r+1 < rows and c < cols:
                	neighbors.append(idx + cols)  	# already done
            	if r+1 < rows and (c+1) < cols:
                	neighbors.append(idx + cols + 1)
        	else:
            	# For odd rows
            	if r-1 >= 0 and (c-1) >= 0:
                	neighbors.append(idx - cols - 1)
            	if r-1 >= 0 and c >= 0:
                	neighbors.append(idx - cols)
            	if r+1 < rows and (c-1) >= 0:
                	neighbors.append(idx + cols - 1)
            	if r+1 < rows and c >= 0:
                	neighbors.append(idx + cols)
       	 
        	# Set adjacency
        	for nb in neighbors:
            	if 0 <= nb < N and nb != idx:
                	A[idx, nb] = 1
                	A[nb, idx] = 1  # symmetric
	return A


# ------------------------------------------------------------
# 2) Build the Hopfield network for channel allocation
# ------------------------------------------------------------

class HopfieldChannelAllocator:
	def __init__(self, adjacency, n_channels=3, alpha=10, beta=10):
    	"""
    	adjacency : NxN adjacency matrix
    	n_channels: number of channels K
    	alpha, beta: penalty coefficients
    	"""
    	self.A = adjacency
    	self.N = adjacency.shape[0]  # number of cells
    	self.K = n_channels
    	self.alpha = alpha
    	self.beta = beta
   	 
    	# We'll store x as (N,K) matrix of ±1 values
    	# Random initialization
    	self.x = np.random.choice([-1, 1], size=(self.N, self.K))
   	 
    	# Construct a weight matrix W (if we want a fully connected Hopfield approach).
    	# But for clarity, we can do direct "energy-based" updates in the update step.
    
	def update_neuron(self, i, k):
    	"""
    	Update a single neuron x[i,k] based on the Hopfield energy gradient.
    	Return the new state (+1 or -1).
    	"""
    	# We'll compute partial derivative of E wrt x[i,k].
    	#  E = alpha * sum_{i} (sum_{k} x_{i,k} - 1)^2
    	#  	+ beta  * sum_{ij in A} sum_{k} x_{i,k} x_{j,k}
   	 
    	# Let sum_x_i = sum_{k} x_{i,k} for cell i.
    	sum_x_i = np.sum(self.x[i,:])
   	 
    	# First part: derivative wrt x_{i,k} from "only one channel per cell"
    	# d/dx_{i,k} of (sum_x_i - 1)^2 = 2(sum_x_i - 1)
    	part1 = 2 * self.alpha * (sum_x_i - 1)
   	 
    	# Second part: derivative wrt x_{i,k} from adjacency penalty
    	# We only care about neighbors j that also have x_{j,k}=+1 or -1
    	# derivative: beta * sum_{j: A[i,j]=1} x_{j,k}
    	# multiply by x_{i,k} as well if we want the sign. Let's be careful:
    	# E contains the term: beta * sum_{i,j in A} sum_{k} x_{i,k} x_{j,k}
    	# partial wrt x_{i,k} => beta * sum_{j: A[i,j]=1} x_{j,k}
   	 
    	part2 = self.beta * np.sum(self.x[self.A[i,:] == 1, k])
   	 
    	# The local field = -(part1 + part2) because Hopfield typically updates
    	# x_{i,k} = sign( - dE/dx_{i,k} )
    	# or we can define a net input h = -(derivative) and x=sign(h).
   	 
    	h = -(part1 + part2)
    	new_state = 1 if h >= 0 else -1
    	return new_state
    
	def run(self, n_iterations=1000):
    	"""
    	Run the Hopfield update for 'n_iterations'.
    	We'll do asynchronous or synchronous updates.
    	"""
    	for it in range(n_iterations):
        	# Option 1: Asynchronous updates in random order
        	i = np.random.randint(0, self.N)
        	k = np.random.randint(0, self.K)
        	self.x[i,k] = self.update_neuron(i,k)
   	 
    	return self.x
    
	def get_assignments(self):
    	"""
    	Convert x from ±1 to a single best channel per cell:
    	We'll pick the channel k with the highest x[i,k].
    	If there's a tie, just pick the first.
    	"""
    	# If x[i,k] is 1 or -1, we can pick the largest
    	# x[i,:] => we pick argmax
    	assigned_channels = np.argmax(self.x, axis=1)
    	return assigned_channels


# ------------------------------------------------------------
# 3) Putting it all together and visualizing
# ------------------------------------------------------------
if __name__ == '__main__':
	# Create adjacency for a small hex layout
	adjacency = create_hex_adjacency(rows=5, cols=5)
	N = adjacency.shape[0]  # total number of cells
    
	# Instantiate Hopfield channel allocator
	n_channels = 3
	allocator = HopfieldChannelAllocator(adjacency, n_channels=n_channels, alpha=10, beta=10)
    
	# Run the network
	allocator.run(n_iterations=10000)
    
	# Get final channel assignment
	channel_assignment = allocator.get_assignments()
    
	# Print results
	print("Final channel assignment for each of the {} cells:".format(N))
	print(channel_assignment)
    
	# Visualize adjacency and assignment
	# We'll place the cells in a simple 2D grid and color them by assigned channel
	# (For a real hex layout, you'd shift every other row, etc.)
    
	# Quick (r, c) coordinates for plotting
	grid_rows = 5
	grid_cols = 5
	coords = []
	for r in range(grid_rows):
    	for c in range(grid_cols):
        	coords.append((r, c))
	coords = np.array(coords)
    
	plt.figure(figsize=(8,6))
	# Plot adjacency edges
	for i in range(N):
    	for j in range(i+1, N):
        	if adjacency[i,j] == 1:
            	plt.plot([coords[i,1], coords[j,1]],
                     	[coords[i,0], coords[j,0]],
                     	'k-', alpha=0.2)
	# Plot cells colored by assigned channel
	colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
	for i in range(N):
    	ch = channel_assignment[i] % len(colors)
    	plt.scatter(coords[i,1], coords[i,0], c=colors[ch], s=200, edgecolors='k')
    	plt.text(coords[i,1], coords[i,0], str(i), ha='center', va='center', color='white', fontsize=8)
    
	plt.title('Hopfield Network Channel Assignment')
	plt.gca().invert_yaxis()
	plt.axis('equal')
	plt.tight_layout()
	plt.show()
