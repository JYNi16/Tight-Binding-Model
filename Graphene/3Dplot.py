# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:15:59 2019

@author: nijinyang
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
a = 3**0.5
x, y = np.mgrid[-3:3:300j, -3:3:300j]
z1 = (3 + 4*(np.cos(0.5*a*y))*(np.cos(1.5*x)) + 2*(np.cos(a*y)))**0.5
z2 = -(3 + 4*(np.cos(0.5*a*y))*(np.cos(1.5*x)) + 2*(np.cos(a*y)))**0.5
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x, y, z2, rstride=1,\
                       cmap=cm.coolwarm, cstride=1, \
                       linewidth=0)
surf = ax.plot_surface(x, y, z1, rstride=1,\
                       cmap=cm.coolwarm, cstride=1, \
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title(r'$Graphe\ band\  structure$')
plt.show()
plt.savefig("3D.pdf")