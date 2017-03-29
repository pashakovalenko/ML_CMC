import numpy as np
from math import sin, cos, exp

def compute_gradient(J, theta, eps=1e-2):
    theta = theta.astype('float64')
    res = np.empty(theta.shape)
    for i in range(theta.shape[0]):
        inp = theta.copy()
        inp[i] += eps
        res[i] = J(inp)
        inp[i] -= 2 * eps
        res[i] -= J(inp)
    res /= 2 * eps
    return res

def my_f(theta):
    x, y, z = theta
    return (x * cos(x * y) + exp(y / z)) / (z ** 0.5 + 1)

def my_f_grad(theta):
    x, y, z = theta
    res = np.empty(3)
    res[0] = (cos(x * y) - x * y * sin(x * y)) / (z ** 0.5 + 1)
    res[1] = (-x ** 2 * sin(x * y) + 1 / z * exp(y / z)) / (z ** 0.5 + 1)
    res[2] = (- (z ** 0.5 + 1) * y / z ** 2 * exp(y / z) - 
              (x * cos(x * y) + exp(y / z)) * 0.5 * z ** (-0.5)) / (z ** 0.5 + 1) ** 2
    return res

def check_gradient():
    points = np.array([[1, 1, 1], [3, 5, 8], [-1, 0, 2], [8, -7, 3]])
    correct = True
    for theta in points:
        a = compute_gradient(my_f, theta)
        b = my_f_grad(theta)
        print(a - b)
        correct &= np.isclose(a, b, rtol=1e-2, atol=1).all()
    if correct:
        print('It\'s OK')
    else:
        raise ArithmeticError('Gradient computing is not correct')