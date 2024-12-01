# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:52:10 2024

@author: 26526
"""

import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from mpmath import polylog
import Berry_curvature as BC
import band_ini.config as cf

miu = 2
sq3 = np.sqrt(3) 

def fermi_dirac(E, t):
    return 1/(np.exp((E-miu)/t) + 1) 


def linear_Hall(t):
    berry = BC.Berry()
    #AFM_m = Ham_AFM_monolayer(12, 12, 12, 0.6, 0.0, 0.0)
    
    xx, yy = cf.xx_h, cf.yy_h
    X,Y = np.meshgrid(xx, yy) 
    
    #honey lattice
    dkx = ((4*sq3/3)*np.pi)/(cf.numk-1)
    dky = 2.0*np.pi/(cf.numk-1)
    
    #square lattice
    #dkx = (2*np.pi)/(cf.numk-1)
    #dky = (2*np.pi)/(cf.numk-1)
    
    j_xy_t = 0
    for i in range(cf.numk):
        for j in range(cf.numk):
            k = np.array([X[i][j], Y[i][j]])
            v, e = berry.ewH(k)
            
            for h in range(4):
            #for h in range(berry.dim):
                f_d = fermi_dirac(e[h], t)
                #k_xy_t += c2_f(n)*np.real(berry.Omega(k,h))
                j_xy_t += f_d * np.real(berry.Omega(k, h))
    
    jxy = (j_xy_t)/(4*np.pi*np.pi) * dkx * dky
    
    return jxy

def plot_xy(data):
    t = []
    kxy = []
    print(data)
    for i in range(len(data)):
        t.append(data[i][0])
        kxy.append(data[i][1])
    
    print("t is:", t, "kxy is:", kxy)
    plt.scatter(t, kxy)
    plt.show()

def cal_Hall():
    T = [round(t,3) for t in np.linspace(0.1, 2, 21)]
    kxy = []
    data = []
    
    for t in T:
        #k = Nonlinear_THE_MMA(t)
        k = linear_Hall(t)
        print("t is:", t, "k is:", k)
        kxy.append(k)
        data.append([t, k])
    print(data)
    #print("J2={} finished !!!".format(J2))
    plot_xy(data)
    


if __name__=="__main__":
    cal_Hall()
