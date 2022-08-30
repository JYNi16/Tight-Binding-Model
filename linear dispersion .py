# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:03:08 2022

@author: 26526
"""
import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

a = np.sqrt(3)

kx = (2*np.pi)/3 
ky = (2*np.pi)/(3*np.sqrt(3))

# x and y range 
x, y = np.mgrid[kx-0.2:kx+0.2:300j, ky-0.2:ky+0.2:300j]

#Energy bands 
z1 = np.sqrt(3 + 4*(np.cos(0.5*a*y))*(np.cos(1.5*x)) + 2*(np.cos(a*y)) + 0.001)
z2 = -z1 

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')

#plot 
surf = ax.plot_surface(x, y, z2, rstride=1,\
                       cmap=cm.coolwarm, cstride=1, \
                       linewidth=0)
surf = ax.plot_surface(x, y, z1, rstride=1,\
                       cmap=cm.coolwarm, cstride=1, \
                       linewidth=0)
ax.view_init(elev=5, azim=45);
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(r'$Graphe\ band\  structure$')
plt.savefig("3D_gap.png", dpi=300)
plt.show()
