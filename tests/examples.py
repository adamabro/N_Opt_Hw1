import numpy as np

def quadratic_example_1(x):
    
    x = np.asarray(x)
    Q = np.array([[1, 0], [0, 1]])
    f = 0.5 * np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q, x)
    h = Q 
    return f, g, h

def quadratic_example_2(x):
    x = np.asarray(x)  
    Q = np.array([[1, 0], [0, 100]])
    f = 0.5 * np.dot(x.T, np.dot(Q, x))  
    g = np.dot(Q, x)
    h = Q 
    return f, g, h

def quadratic_example_3(x):
    x = np.asarray(x)
    
    Q = np.array([[75.25, -(np.sqrt(3)*24.75)], [-(np.sqrt(3)*24.75), 25.75]]) 
    
    f = 0.5 * np.dot(x.T, np.dot(Q,x))
    g = np.dot(Q, x)
    
    
    t = 1e-8
    h = Q + t + np.eye(Q.shape[0])
    return f, g, h

def rosenbrock(x):
    
    x = np.array(x)
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([-400 * (x[1] - x[0]**2) * x[0] - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)])
    h = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
    return f, g, h

def linear_function(x):
    x = np.array(x)
    a = np.array([1, 2])
    f = a @ x
    g = a
    h = np.zeros((2, 2))
    t = 1e-8
    h = h + t + np.eye(h.shape[0])
    return f, g, h

def boyds_function(x):
    x = np.array(x)
    f = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = np.array([np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)])
    h = np.array([[np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1),
                   3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)],
                  [3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1),
                   9*np.exp(x[0] + 3*x[1] - 0.1) + 9*np.exp(x[0] - 3*x[1] - 0.1)]])
    return f, g, h
