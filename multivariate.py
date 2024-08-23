import numpy as np

def gradient(f, x):
    grad = np.zeros_like(x, dtype=float)
    h = 1e-5  # small step for numerical differentiation
    for i in range(len(x)):
        x_step = np.copy(x)
        x_step[i] += h
        grad[i] = (f(x_step) - f(x)) / h
    return grad

def hessian(f, x):
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    h = 1e-5  # small step for numerical differentiation
    for i in range(n):
        for j in range(n):
            x_ij = np.copy(x)
            x_ij[i] += h
            x_ij[j] += h
            hess[i, j] = (f(x_ij) - f(x)) / h
    return hess

def newton_multivariate(f, x0, epsilon):
    x = np.array(x0, dtype=float)
    n_iter = 0
    
    while True:
        grad_f = gradient(f, x)
        hess_f = hessian(f, x)
        if np.linalg.det(hess_f) == 0:  # Check if Hessian is singular
            print("Hessian matrix is singular. Cannot proceed.")
            return None
        
        # Newton's update: x_{n+1} = x_n - H^{-1} * grad_f
        delta_x = np.linalg.solve(hess_f, grad_f)
        x_new = x - delta_x
        
        # Stop if the update is small enough
        if np.linalg.norm(x_new - x) < epsilon:
            break
        
        x = x_new
        n_iter += 1
    
    print('After', n_iter, 'iterations, the root is', x)
    return x

# Example usage:

# Define a multivariate function
def f(x):
    return x[0]**2 + x[1]**2 + 3 * x[0] * x[1]

# Initial guess
x0 = [1, 1]

# Set tolerance
epsilon = 1e-6

# Run the Newton's method
root = newton_multivariate(f, x0, epsilon)
print("Root found:", root)