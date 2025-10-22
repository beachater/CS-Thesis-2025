"""
The benchmark functions
"""

import math
import numpy as np


def get_function_by_name(name):
    # Multimodal
    if name == 'Ackley': return Ackley
    if name == 'Griewank': return Griewank
    if name == 'Rastrigin': return Rastrigin
    if name == 'SchafferN2': return SchafferN2
    if name == 'SchafferN4': return SchafferN4
    if name == 'Shubert': return Shubert
    if name == 'Eggholder': return Eggholder
    if name == 'Levy': return Levy
    if name == 'Schwefel': return Schwefel
    # Fixed dimension
    if name == 'HolderTable': return HolderTable


def Ackley(x, lb=-32.768, ub=32.768):
    s1 = np.sum(np.square(x))
    s2 = np.sum(np.cos(2 * np.pi * x))
    d = len(x)
    y = -20 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20 + np.e
    return y, lb, ub


def Griewank(x, lb=-600., ub=600.):
    d = len(x)
    s1 = np.sum(np.square(x)) / 4000
    s2 = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    y = s1 - s2 + 1
    return y, lb, ub


def Rastrigin(x, lb=-5.12, ub=5.12):
    d = len(x)
    y = 10 * d + np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x))
    return y, lb, ub


def SchafferN2(x, lb=-100., ub=100.):
    if len(x) != 2:
        raise ValueError("Schaffer N.2 is defined for 2 dimensions only.")
    numerator = np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5
    denominator = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    y = 0.5 + numerator / denominator
    return y, lb, ub


def SchafferN4(x, lb=-100., ub=100.):
    if len(x) != 2:
        raise ValueError("Schaffer N.4 is defined for 2 dimensions only.")
    numerator = np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5
    denominator = (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2
    y = 0.5 + numerator / denominator
    return y, lb, ub


def Shubert(x, lb=-10., ub=10.):
    if len(x) != 2:
        raise ValueError("Shubert is defined for 2 dimensions only.")
    sum1 = np.sum([i * np.cos((i + 1) * x[0] + i) for i in range(1, 6)])
    sum2 = np.sum([i * np.cos((i + 1) * x[1] + i) for i in range(1, 6)])
    y = sum1 * sum2
    return y, lb, ub


def Eggholder(x, lb=-512., ub=512.):
    if len(x) != 2:
        raise ValueError("Eggholder is defined for 2 dimensions only.")
    term1 = -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0] / 2 + 47)))
    term2 = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
    y = term1 + term2
    return y, lb, ub


def Levy(x, lb=-10., ub=10.):
    """
    Standard Levy Function N.13
    Global minimum: f(x) = 0 at x_i = 1 for all i
    """
    x = np.array(x)
    w = 1 + (x - 1) / 4

    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

    y = term1 + term2 + term3
    return y, lb, ub



def Schwefel(x, lb=-500., ub=500.):
    x = np.array(x)
    y = 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return y, lb, ub


def HolderTable(x, lb=-10., ub=10.):
    if len(x) != 2:
        raise ValueError("Holder Table is defined for 2 dimensions only.")
    y = -np.abs(
        np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)))
    return y, lb, ub
