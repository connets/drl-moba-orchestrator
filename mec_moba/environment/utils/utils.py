import numpy as np


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    valueScaled = float(value - leftMin) / float(leftSpan)

    return rightMin + (valueScaled * rightSpan)


def gini(s):
    if (2 * len(s) * sum(s)) > 0:
        return sum([abs(s[i] - s[j]) for i in range(len(s)) for j in range(len(s))]) / (2 * len(s) * sum(s))
    return 0


def sig(x, x0=0, L=1, k=1):
    return L / (1 + np.exp(-k * (x - x0)))


def exp_lin(x):
    if x < 0:
        return np.exp(x)
    return x + 1
