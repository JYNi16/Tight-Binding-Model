# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:37:20 2024

@author: curry
"""

import numpy as np 
from BHZ_model import BHZ 
from band_ini import config as cf

Ham = BHZ(-2.1)

A=1
B=1
delta=1

def test_model(k):
    kx, ky = k
    
    return 1*np.sin(kx)*cf.sx+1*np.sin(ky)*cf.sy+(delta-4*B*(np.sin(kx/2)**2)-4*B*np.sin(ky/2)**2)*cf.sz;

def H(k):
    
    return test_model(k)

def ewH(k):
    e,w=np.linalg.eigh(H(k))
    e,w=np.linalg.eigh(H(k))
    w0 = w[:, np.argsort(np.real(e))[0]]
    w1 = w[:, np.argsort(np.real(e))[1]]
    e = np.sort(np.real(e))
        
    return w0, e[0]

def Wcc():
    xx = np.linspace(-np.pi, np.pi, 161)
    yy = np.linspace(-np.pi, np.pi, 101)
    for kx in xx: 
        Ds = np.zeros((1,1))
        vD = 1
        for b in range(len(yy)-1):
            ky = yy[b]
            w, e = ewH([kx, ky])
            
            VN = w 
            ky2 = yy[b+1]
            w, e = ewH([kx, ky2])
            VM = w
            
            Ds[0,0] = np.dot(VN.conj().T, VM)
            
            vD *= Ds
            
            print("vD is:",Ds)
        
        k_sita = np.real(1.j*(np.log(np.linalg.eigh(vD)[0]))/(2*np.pi))
        
        print("k_sita is:", k_sita)
    
            


Wcc()