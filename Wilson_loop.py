# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:37:20 2024

@author: curry
"""

import numpy as np 
from BHZ_model import BHZ 
from band_ini import config as cf
import matplotlib.pyplot as plt

Ham = BHZ(-2.5)

A=1
B=1
delta=1

Nband = 1

def test_model(k):
    kx, ky = k
    
    return 1*np.sin(kx)*cf.sx+1*np.sin(ky)*cf.sy+(delta-4*B*(np.sin(kx/2)**2)-4*B*np.sin(ky/2)**2)*cf.sz;

def H(k):
    
    return test_model(k)

def ewH(k):
    e,w=np.linalg.eigh(H(k))
    
    s_idx = np.argsort(e)
    s_e = e[s_idx]
    s_w = w[:, s_idx]       
    return s_w

def Vmn(w1, w2, Ds): 
    for i in range(Nband):
        for j in range(Nband):
            Ds[i,j] = np.dot(w1[:,i].conj().T, w2[:,j])
    return Ds
            

def Wcc():
    xx = np.linspace(-np.pi, np.pi, 161)
    yy = np.linspace(-np.pi, np.pi, 101)
    k_sita = []
    for i in range(len(xx)): 
        Ds = np.zeros((Nband, Nband), dtype=complex)
        vD = np.ones((Nband, Nband), dtype=complex)
        for b in range(len(yy)-1):
            ky = yy[b]
            VN = ewH([xx[i], ky])
            
            ky2 = yy[b+1]
            VM = ewH([xx[i], ky2])
            
            Ds = Vmn(VN, VM, Ds) 
            vD = np.dot(vD, Ds)
        
        k_sita.append(np.real(1.j*(np.log(np.linalg.eig(vD)[0])))/(2*np.pi))
    
    return xx, k_sita[::-1]


def plot_wcc():
    xx, k_sita = Wcc()
    
    font = {'family': "Times New Roman", "weight":"normal", "size":24,}
    fig = plt.figure(figsize=(10,8))
    
    plt.scatter(xx, k_sita, c= "none", s= 75, marker = "o", edgecolors="r")
    #plt.scatter(xx, yy, c=Z)
    #C=plt.contour(X,Y,Z,10,colors='black',linewidths=0.1)
    plt.yticks(fontproperties='Times New Roman', fontsize = 20)
    plt.xticks(fontproperties='Times New Roman', fontsize = 20)
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-0.5,0.5)
    plt.xlabel(r"$k_{x}$", font)
    plt.ylabel(r"$Wcc$", font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
            


if __name__=="__main__":
    plot_wcc()