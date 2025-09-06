"""
The benchmark functions

"""

import math
import random
import numpy as np


# Benchmark functions are used to evaluate the fitness of candidate solutions (antibodies), 
# which are sampled from within the function’s defined bounds. 
# The fitness value reflects how close each solution is to the global optimum, guiding the algorithm’s 
# search toward better solutions.

def get_function_by_name(name):
    # Unimodal
    if name == 'Sphere': return Sphere
    if name == 'BentCigar': return BentCigar
    if name == 'PowellSum': return PowellSum
    if name == 'ChungReynolds': return ChungReynolds
    if name == 'Csendes': return Csendes
    if name == 'Holzman_02': return Holzman_02
    if name == 'SchumerSteiglitz_02': return SchumerSteiglitz_02
    if name == 'Schwefel_1_2': return Schwefel_1_2
    if name == 'Schwefel_2_20': return Schwefel_2_20

    # Multimodal
    if name == 'Ackley': return Ackley #fcsa
    if name == 'Griewank': return Griewank #fcsa
    if name == 'InvertedCosineWave': return InvertedCosineWave
    if name == 'Rastrigin': return Rastrigin #fcsa
    if name == 'Pathological': return Pathological
    if name == 'Shubert_06': return Shubert_06 #fcsa
    if name == 'Quartic': return Quartic
    if name == 'DeflectedCorrugatedSpring': return DeflectedCorrugatedSpring
    if name == 'XinSheYang_07': return XinSheYang_07

    # Fixed dimension
    if name == 'Eggcrate': return Eggcrate #fcsa
    if name == 'HolderTable': return HolderTable #fcsa
    if name == 'BohachevskyNO_2': return BohachevskyNO_2
    if name == 'Foxholes': return Foxholes
    if name == 'Davis': return Davis
    if name == 'ModifiedSchaffer_01': return ModifiedSchaffer_01 #fcsa
    if name == 'ModifiedSchaffer_02': return ModifiedSchaffer_02  #fcsa
    if name == 'Price_02': return Price_02
    if name == 'Tsoulos': return Tsoulos


def Sphere(x, lb=-10., ub=10.):
    """f(x) = 0, x = (0, ..., 0)"""
    y = 0.
    for i in range(len(x)):
        y = y + x[i] ** 2
    return y, lb, ub


def PowellSum(x, lb=-1., ub=1.):
    """f(x) = 0, x = (0, ..., 0)"""
    y = 0.
    for i in range(len(x)):
        y = y + np.fabs(x[i]) ** (i + 2)
    return y, lb, ub


def Schwefel_1_2(x, lb=-10., ub=10.):
    """
    Double-Sum or Rotated Hyper-Ellipsoid Function
    f(x) = 0, x = (0, ..., 0)
    """
    y = 0.
    for i in range(len(x)):
        s = 0.
        for j in range(i + 1):
            s = s + x[j]
        y = y + s ** 2
    return y, lb, ub


def Schwefel_2_20(x, lb=-100., ub=100.):
    """f(x) = 0, x = (0, ..., 0)"""
    y = 0.
    for i in range(len(x)):
        y = y + np.fabs(x[i])
    return y, lb, ub


def ChungReynolds(x, lb=-100., ub=100.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d):
        y += np.power(x[i], 2)
    y = np.power(y, 2)
    return y, lb, ub


def Csendes(x, lb=-100., ub=100.):
    """
    or EX3 or Infinity function
    f(x) = 0, x = (0, ..., 0)
    """
    d = len(x)
    y = 0.0
    for i in range(d):
        if x[i] == 0:
            y += 0
        else:
            y += x[i]**6*(2+np.sin(1/x[i]))
    return y, lb, ub


def Holzman_02(x, lb=-10., ub=10.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d):
        y += (i+1)*x[i]**4
    return y, lb, ub


def SchumerSteiglitz_02(x, lb=-100., ub=100.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d):
        y += np.power(x[i], 4)
    return y, lb, ub


def BentCigar(x, lb=-100., ub=100.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    y1 = 0.0
    for i in range(d):
        if i == 0:
            y += x[0]**2
        else:
            y1 += x[i]**2
    y += y1*10**6
    return y, lb, ub


def Ackley(x, lb=-32.768, ub=32.768):
    """
    or Ackley's Path
    f(x) = 0, x = (0, ...,0)
    """
    y = 0.
    s1 = 0.
    s2 = 0.
    for i in range(len(x)):
        s1 = s1 + x[i] ** 2
        s2 = s2 + np.cos(2 * math.pi * x[i])
    if len(x) > 0:
        y = -20 * np.exp(-0.2 * np.sqrt(s1 / len(x))) - np.exp(s2 / len(x)) + 20 + np.e
    return y, lb, ub


def Griewank(x, lb=-600., ub=600.):
    """f(x) = 0, x = (0, ..., 0)"""
    s1 = 0.
    s2 = 1.
    for i in range(len(x)):
        s1 = s1 + x[i] ** 2
        s2 = s2 * np.cos(x[i] / np.sqrt(i + 1))
    return s1 / 4000 - s2 + 1, lb, ub


def Rastrigin(x, lb=-5.12, ub=5.12):
    """f(x) = 0, x = (0, ..., 0)"""
    y = 10 * len(x)
    for i in range(len(x)):
        y = y + x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
    return y, lb, ub


def XinSheYang_07(x, lb=-10., ub=10.):
    """
    or Stochastic
    f(x) = 0, x = (1/i, ...)
    """
    d = len(x)
    y = 0
    for i in range(d):
        y += random.random() * abs(x[i] - 1 / (i + 1))
    return y, lb, ub


def Quartic(x, lb=-1.28, ub=1.28):
    """
    or Modified 4th De Jong's
    f(x) = 0, x = (0, ..., 0)
    """
    y = 0.
    for i in range(len(x)):
        y = y + (i + 1) * x[i] ** 4
    y = y + random.random()
    return y, lb, ub


def InvertedCosineWave(x, lb=-5., ub=5.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d-1):
        y += -np.exp(-(x[i]**2+x[i+1]**2+0.5*x[i]*x[i+1])/8) * \
            np.cos(4*np.sqrt(x[i]**2+x[i+1]**2+0.5*x[i]*x[i+1]))
    y += d-1
    return y, lb, ub


def Pathological(x, lb=-100., ub=100.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d - 1):
        y1 = np.power(
            np.sin(np.sqrt(100 * np.power(x[i], 2) + np.power(x[i + 1], 2))), 2) - 0.5
        y2 = 1 + 0.001 * \
            np.power(np.power(x[i], 2) - 2 * x[i] *
                     x[i + 1] + np.power(x[i + 1], 2), 2)
        y += 0.5+y1/y2
    return y, lb, ub


def Shubert_06(x, lb=-10., ub=10.):
    """f(x) = 0, x = (0, ..., 0)"""
    d = len(x)
    y = 0.0
    for i in range(d-1):
        y += 0.5+(np.power(np.sin(np.sqrt(np.power(x[i], 2)+np.power(x[i+1], 2))), 2)-0.5)/np.power(
            1+0.001*(np.power(x[i], 2)+np.power(x[i+1], 2)), 2)
    return y, lb, ub


def DeflectedCorrugatedSpring(x, lb=-0., ub=10.):
    """f(x) = 0, x = (5, ..., 5)"""
    d = len(x)
    y1 = 0.0
    a = 5
    k = 5
    for i in range(d):
        y1 += (x[i]-a)**2
    y = 1+0.1*y1-np.cos(k*np.sqrt(y1))
    return y, lb, ub


def HolderTable(x, lb=-10., ub=10.):
    """
    Multimodal, Dimensions: 2
    f(x) = -19.2085, x = (8.05502, 9.66459), (8.05502, -9.66459), (-8.05502, 9.66459),
    (-8.05502, -9.66459)
    """
    y = 0.
    if len(x) > 0:
        y = -np.fabs(
            np.sin(x[0]) * np.cos(x[1]) * np.exp(np.fabs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))
    return y, lb, ub


def Eggcrate(x, lb=-5., ub=5.):
    """
    Multimodal, Dimensions: 2
    f(x) = 0
    """
    y = 0.
    if len(x) > 0:
        y = x[0] ** 2 + x[1] ** 2 + 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)
    return y, lb, ub


def BohachevskyNO_2(x, lb=-100., ub=100.):
    """
    Multimodal, Dimensions: 2
    f(x) = 0
    """
    y = 0.
    if len(x) > 0:
        y = x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * np.cos(3 * np.pi * x[0]) * np.cos(4 * np.pi * x[1]) + 0.3
    return y, lb, ub


def Foxholes(x, lb=-65.53, ub=65.53):
    """
    Multimodal, Dimensions: 2
    f(x) = 1
    """
    y = 0.
    a = [[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
         [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
    if len(x) > 0:
        for j in range(25):
            s = 0.
            for i in range(2):
                s = s + (x[i] - a[i][j]) ** 6
            y = y + 1 / (j + 1 + s)
        y = 1 / (1 / 500 + y)
    return y, lb, ub


def Davis(x, lb=-100., ub=100.):
    """
    Multimodal, dimension: 2
    f(x) = 0
    """
    d = len(x)
    y = 0.0
    if d == 2:
        y = np.power(np.power(x[0], 2)+np.power(x[1], 2), 0.25)*(1+np.power(
            np.sin(50*np.power(3*np.power(x[0], 2)+np.power(x[1], 2), 0.1)), 2))
    return y, lb, ub


def ModifiedSchaffer_01(x, lb=-10., ub=10.):
    """
    multimodal, dimension: 2
    f(x) = 0
    """
    d = len(x)
    y = 0.0
    if d == 2:
        y = 0.5+((np.sin(x[0]**2+x[1]**2))**2-0.5) / \
            (1+0.001*(x[0]**2+x[1]**2))**2
    return y, lb, ub


def ModifiedSchaffer_02(x, lb=-100., ub=100.):
    """
    multimodal, dimension: 2,
    f(x) = 0
    """
    d = len(x)
    y = 0.0
    if d == 2:
        y = 0.5+((np.sin(x[0]**2-x[1]**2))**2-0.5) / \
            (1+0.001*(x[0]**2+x[1]**2))**2
    return y, lb, ub


def Price_02(x, lb=-10., ub=10.):
    """
    or Periodic
    multimodal, dimension: 2
    f(x) = 0
    """
    d = len(x)
    y = 0.0
    if d == 2:
        y = 0.1+np.power(np.sin(x[0]), 2)+np.power(np.sin(x[1]), 2) - \
            0.1*np.exp(-np.power(x[0], 2)-np.power(x[1], 2))
    return y, lb, ub


def Tsoulos(x, lb=-1., ub=1.):
    """
    multimodal, dimension: 2
    f(x) = 0
    """
    d = len(x)
    y = 0.0
    if d == 2:
        y = x[0]**2+x[1]**2-math.cos(18*x[0])-math.cos(18*x[1])
    y += 2
    return y, lb, ub