import numpy as np

# Define alpha and transition matrix
np.random.seed(101)
alpha = 0.2
M = np.array([
    [0, 1/2, 0, 0, 1/2, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1/2, 0, 1/2],
    [1/2, 0, 0, 1/2, 0, 0, 0],
    [0, 0, 0, 1/2, 1/2, 0, 0],
    [0, 1/2, 0, 0, 1/2, 0, 0]
])

# Initialize page rank to random values
p_t = np.random.rand(7, 1)
p_t1 = (alpha * np.eye(len(M)) + (1-alpha)*M).T.dot(p_t)

# Keep calculating until there is no change
while np.any(np.abs(p_t - p_t1) > 0):
    p_t = p_t1
    p_t1 = (alpha * np.eye(len(M)) + (1-alpha)*M).T.dot(p_t)

