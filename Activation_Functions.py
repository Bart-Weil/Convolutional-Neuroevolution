#for activation functions

import math

def binstep(x):
    if x<0:

        return 0
    else:

        return 1

def linear(x):

    return x

def sigmoid(x):

    return (1+math.e**(-x))**-1

def tanh(x):

    return 2*(1+math.e**(-2*x))**-1 -1

def relu(x):

    return max(0,x)

def lrelu(x):

    if x>=0:

        return x

    else:
        return 0.01*x

def elu(x):
    if x<0:

        return math.e**x-1
    else:

        return x

def swish(x):

    return x * (1+math.e**(-x))**-1

def gauss(x):

    return math.e**(-x**2)

def absolute(x):

    if x>=0:

        return x
    else:
        return -x
