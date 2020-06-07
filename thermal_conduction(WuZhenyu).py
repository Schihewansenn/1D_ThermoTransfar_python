import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import *
from mpl_toolkits.mplot3d import Axes3D

alpha = 8.9 # parameters derived from physical properties of the material
x = np.arange(0,30.1,0.1) # length of rod
t = np.arange(0.0,100.0,0.1) # length of time imitated
T = np.cos(x)+np.sin(0.6*x) # initial temperature distribution

def f2(T, t):
    arr = np.zeros(len(T+2))
    arr[1:-1] = np.diff(T, 2)
    return alpha * arr

def f1(T):
    states = odeint(f2,y0=T,t=t)
    return states

res = f1(T)
X,T = np.meshgrid(x,t)

ax= plt.subplot(111,projection='3d')
ax.plot_surface(X,T,res,cmap=cm.coolwarm)
plt.xlabel('Length')
plt.ylabel('Time')
plt.show()