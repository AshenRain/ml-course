import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    n = data.shape[0]
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    
    for i in range(num_steps):
        x = data @ x
        x /= np.linalg.norm(x)
        
    eigenvalue = x @ data @ x
    eigenvector = x
    
    return float(eigenvalue), eigenvector
