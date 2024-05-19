import numpy as np

def backtrack_search (f, grad, x0, d, g, c1=0.01, alpha=0.5):
    
    s = 1
    
    while f(x0 + s * d) > f(x0) + c1 * s * np.dot(g,d):
        s = s * alpha 
    return s

def newton_method (f, grad, hess, x0, obj_tol, param_tol, max_iter, c1=0.01, alpha=0.5):
    
    x = np.array(x0, dtype = float)
    
    path = [x]
    
    values = [f(x)]
    
    print(f"Newton method Iterations")
    for i in range(max_iter):
        g = grad(x)
        h = hess(x)
        if h is None:
            raise ValueError('Hessian matrix required for Newton\'s method.')
        
        d = - np.linalg.solve(h, g)
        
        a_i = backtrack_search(f, grad, x, d, g, c1, alpha)
        
        x1 = x + a_i * d
        
        print(x1)
        
        path.append(x1)
        values.append(f(x1))
        
        print(f"Iteration {i}: x = {x1}, f(x) = {f(x1)}")
        
        if np.abs(f(x1) - f(x)) < obj_tol or np.linalg.norm(x1-x) < param_tol:
            return x1, f(x1), True, path, values
        
        x = x1
    print("Newton Method: Failed to converge")
    return x, f(x), False, path, values

def gradient_descent(f, grad, x0, obj_tol, param_tol, max_iter, c1=0.01, alpha=0.5):
    x = np.array(x0, dtype=float)
    path = [x]
    values = [f(x)]
    for i in range(max_iter):
        g = grad(x)
        a_i = backtrack_search(f, grad, x, -g, g, c1, alpha)
        x1 = x - a_i * g
        path.append(x1)
        values.append(f(x1))
        
        
        print(f"Iteration {i}: x = {x1}, f(x) = {f(x1)}, step size = {a_i}, grad = {g}")
        
        if np.abs(f(x1) - f(x)) < obj_tol or np.linalg.norm(x1 - x) < param_tol:
            print("Converged")
            return x1, f(x1), True, path, values
        x = x1
    
    print("Gradient Descent: Failed to converge")
    return x, f(x), False, path, values

def minimize(f, x0, obj_tol, param_tol, max_iter, method='gradient_descent', grad=None, hess=None):
    
    if method == 'gradient_descent':
        
        if grad is None:
            raise ValueError('Gradient function is required for Gradient Descent.')
        return gradient_descent(f, grad, x0, obj_tol, param_tol, max_iter)
    
    elif method == 'newton':
        
        if grad is None or hess is None:
            raise ValueError('Both gradient and Hessian functions are required for Newton\'s method.')
        return newton_method(f, grad, hess, x0, obj_tol, param_tol, max_iter)
    
    else:
        raise ValueError('Invalid method provided')
        
            
        
    