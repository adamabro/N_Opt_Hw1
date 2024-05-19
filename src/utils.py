import matplotlib.pyplot as plt
import numpy as np

def plot_contour_with_paths(func, limits, paths=None):
    
    x_min, x_max, y_min, y_max = limits
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.array([[func([xi, yi]) for xi in x] for yi in y])
    plt.contour(X, Y, Z, levels=20, cmap='viridis')

    if paths:
        for path, name in paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=name, marker='o')
        plt.legend()

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Contours with Paths')
    plt.grid(True)
    plt.show()


def plot_function_values(*methods):
    
    for values, name in methods:
        plt.plot(range(len(values)), values, label=name)

    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.show()

