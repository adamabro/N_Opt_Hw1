import sys
import os
import unittest
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from unconstrained_min import minimize
from examples import quadratic_example_1, quadratic_example_2, quadratic_example_3, rosenbrock, linear_function, boyds_function
from utils import plot_contour_with_paths, plot_function_values

class TestUnconstrainedMinimization(unittest.TestCase):
    

    def setUp(self):
        
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 1000
        self.x0 = [1, 1]

    def test_quadratic_example_1(self):
        
        f = lambda x: quadratic_example_1(x)[0]
        grad = lambda x: quadratic_example_1(x)[1]
        hess = lambda x: quadratic_example_1(x)[2]

        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

        
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )
    
    
    def test_quadratic_example_2(self):
       
        f = lambda x: quadratic_example_2(x)[0]
        grad = lambda x: quadratic_example_2(x)[1]
        hess = lambda x: quadratic_example_2(x)[2]

        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

        
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )
    
    def test_quadratic_example_3(self):
       
        f = lambda x: quadratic_example_3(x)[0]
        grad = lambda x: quadratic_example_3(x)[1]
        hess = lambda x: quadratic_example_3(x)[2]

        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

        
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )

    
    def test_rosenbrock(self):
        f = lambda x: rosenbrock(x)[0]
        grad = lambda x: rosenbrock(x)[1]
        hess = lambda x: rosenbrock(x)[2]

        result_gd = minimize(f, [-1,2], self.obj_tol, self.param_tol, 10000, method='gradient_descent', grad=grad)
        result_nt = minimize(f, [-1,2], self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

       
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )
    
    def test_linear(self):
       
        f = lambda x: linear_function(x)[0]
        grad = lambda x: linear_function(x)[1]
        hess = lambda x: linear_function(x)[2]

        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

        
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )
    
    def test_boyds(self):
       
        f = lambda x: boyds_function(x)[0]
        grad = lambda x: boyds_function(x)[1]
        hess = lambda x: boyds_function(x)[2]

        result_gd = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='gradient_descent', grad=grad)
        result_nt = minimize(f, self.x0, self.obj_tol, self.param_tol, self.max_iter, method='newton', grad=grad, hess=hess)

        self.assertTrue(result_gd[2], "Gradient Descent failed")
        self.assertTrue(result_nt[2], "Newton's Method failed")

        
        limits = [-2, 2, -2, 2]
        plot_contour_with_paths(f, limits, paths=[(result_gd[3], "Gradient Descent"), (result_nt[3], "Newton's Method")])
        
        
        plot_function_values(
            (result_gd[4], "Gradient Descent"),
            (result_nt[4], "Newton's Method")
        )
    
     
if __name__ == '__main__':
    unittest.main()
