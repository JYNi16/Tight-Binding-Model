# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:44:15 2022

@author: jyni
"""
import numpy as np 
import math 
import matplotlib.pyplot as plt

M = (2 * np.pi) / 3
K = (2*np.pi)/(3*np.sqrt(3)) 


x = np.linspace(0, M)               
x2 = np.linspace(-K, 0)
x3 = np.linspace(-M, 0)

#along Gamma to K points path 
y_p = np.sqrt(1 + 4 * np.cos(1.5 * x) * np.cos(0.5 * x) + 4 * np.power(np.cos(0.5*x), 2))
y_n = -np.sqrt(1 + 4 * np.cos(1.5 * x) * np.cos(0.5 * x) + 4 * np.power(np.cos(0.5*x), 2))

#along K to M points path 
y2_p = (1 - 2*np.cos( (np.sqrt(3)/2) *x2))
y2_n = -(1 - 2*np.cos( (np.sqrt(3)/2) *x2))

#along M to Gamma points path
y3_p = -np.sqrt(5 + 4*np.cos(1.5 * x3))
y3_n = np.sqrt(5 + 4*np.cos(1.5 * x3))


plt.plot(x, y_n, "b")
plt.plot(x, y_p, "red")

plt.plot(x2+M+K, y2_p, "b")
plt.plot(x2+M+K, y2_n, "red")

plt.plot(x3+2*M+K, y3_p, "b")
plt.plot(x3+2*M+K, y3_n, "red")

plt.axhline(y=0,ls=":",c="blue")#添加水平直线

plt.xlim(0, M+2*K)
plt.ylim(-3.5, 3.5)

plt.xlabel('$Kpoints$', fontproperties = 'Times New Roman',size = 16)

xticks  = [0, M, M+K, M+2*K]

xticks_label = [r'$\Gamma$','$K$','$M$',r'$\Gamma$']
plt.xticks(xticks,xticks_label, fontproperties = 'Times New Roman', size = 14)

plt.show() 


