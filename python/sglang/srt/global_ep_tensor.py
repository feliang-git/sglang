import numpy as np
import torch
# Set a seed for reproducibility
np.random.seed(42)

# Create a global tensor variable for testing
# 61 rows, each row has a random permutation of integers from 0 to 255
EP_LOAD_TENSOR = torch.tensor(
    np.array([
        np.random.permutation(256)
        for _ in range(61)
    ]), 
    dtype=torch.int64
)
# print(EP_LOAD_TENSOR)