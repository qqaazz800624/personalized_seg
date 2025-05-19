#%%

import numpy as np
import math

def D(t):
    return math.ceil(math.log(4097, 2*t))

print(D(8))
print(D(9))
print(D(10))
print(D(14))
print(D(33))

#%%

def block(t):
    return (1+0.032*t)

print(block(8))
print(block(9))
print(block(10))
print(block(14))
print(block(33))

#%%

def T(t):
    return D(t) * block(t)

print(T(8))
print(T(9))
print(T(10))
print(T(14))
print(T(33))


#%%







#%%