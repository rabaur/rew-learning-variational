import numpy as np

def sigmoid(x: float) -> float:
    """
    Compute numerically stable sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid of x
    """
    if x < 0:
        z = np.exp(x)
        return z / (1 + z)
    else:
        return 1.0 / (1.0 + np.exp(-x))