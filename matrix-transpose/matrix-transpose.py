import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    n_rows = len(A)
    n_cols = len(A[0])
    
    # Create empty output array with swapped dimensions (M, N)
    transposed = np.zeros((n_cols, n_rows), dtype=type(A[0][0]))
    
    for i in range(n_rows):
        for j in range(n_cols):
            transposed[j, i] = A[i][j]
            
    return transposed