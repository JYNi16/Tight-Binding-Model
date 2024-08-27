# -*- coding: utf-8 -*-
"""
Spyder Editor

Python script to calculate the Kane-mele band
Author: Jinyang Ni
"""
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from band_ini import config as cf


#spin exchange parameters 

#define the Hamiltonian 

class Honeycomb():
    
    def __init__(self, t1, D1, J2, A):
        self.H =np.zeros((2,2), dtype=complex)
        self.t1 = t1 
        self.D1 = 0.1
        self.J2 = 0
        self.A =  0
        self.A0 = 0.0
        self.tt = -0.0 # titling term
    
    def model(self, k):
        kx, ky = k
        gk = np.exp(1.j*k.dot(cf.a1)) + np.exp(1.j*k.dot(cf.a2)) + np.exp(1.j*k.dot(cf.a3))
        self.H[0,1] = self.t1 * gk
        self.H[1,0] = self.t1 * gk.conj()
    
        #add NNN conj hopping term 
        dk = 2*np.sin(k.dot(cf.d1)) + np.sin(k.dot(cf.d2)) + np.sin(k.dot(cf.d3))
        Hd = self.D1*dk * cf.sz
        jk = self.J2*(2*(np.cos(k.dot(cf.d1)) + np.cos(k.dot(cf.d2)) + np.cos(k.dot(cf.d3))) - 6)
        H2 = jk * cf.sz + self.A * cf.sz + self.A0 * cf.s0 + self.tt*(kx + ky) * cf.s0
        return np.array(self.H + Hd + H2)


class Zigzag():
    
    def __init__(self):
        self.H =np.zeros((4,4), dtype=complex)
        self.t1 = -1
        self.D1 = 0.1
        self.J2 = 0.1
        self.M1 =  -0.2
        self.M2 =  -0.2
        self.M3 = 0.2
        self.M4 = 0.2
        self.tt = -0.1 # titling term
    
    def model(self, k):
        kx, ky = k
        g1k = self.t1 * (np.exp(1.j*k.dot(cf.az2)) + np.exp(1.j*k.dot(cf.az3)))
        g2k = self.t1*np.exp(1.j*k.dot(cf.az1))
        
        f1k = self.D1*np.sin(k.dot(cf.dz2))
        f2k = self.D1*(np.sin(k.dot(cf.dz1)) + np.sin(k.dot(cf.dz3)))
        
        self.H[0,1] = g1k
        self.H[1,0] = g1k.conj()
        
        self.H[0,2] = g2k 
        self.H[2,0] = g2k.conj()
        
        self.H[1,3] = g2k.conj()
        self.H[3,1] = g2k 
        
        self.H[2,3] = g1k.conj()
        self.H[3,2] = g1k
        
        self.H[0,0], self.H[1,1], self.H[2,2], self.H[3,3] = self.M1 + f1k, self.M2-f1k, self.M3-f1k, self.M4+f1k
        
        self.H[0,3], self.H[1,2], self.H[2,1], self.H[3,0] = f2k, -f2k, -f2k, f2k
        
        #adding titlting term
        Htt = self.tt*(kx) * np.eye(4, dtype=complex)
        
        return np.array(self.H + Htt)

class stripe():
    
    def __init__(self):
        self.H =np.zeros((4,4), dtype=complex)
        self.t1 = -1
        self.D1 = 0.0
        self.J2 = 0.1
        self.M1 =  0.2
        self.M2 =  -0.2
        self.M3 = 0.2
        self.M4 = -0.2
        self.tt = -0.0 # titling term
    
    def model(self, k):
        kx, ky = k
        g1k = self.t1 * (np.exp(1.j*k.dot(cf.az2)) + np.exp(1.j*k.dot(cf.az3)))
        g2k = self.t1*np.exp(1.j*k.dot(cf.az1))
        
        f1k = self.D1*np.sin(k.dot(cf.dz2))
        f2k = self.D1*(np.sin(k.dot(cf.dz1)) + np.sin(k.dot(cf.dz3)))
        
        self.H[0,1] = g1k
        self.H[1,0] = g1k.conj()
        
        self.H[0,2] = g2k 
        self.H[2,0] = g2k.conj()
        
        self.H[1,3] = g2k.conj()
        self.H[3,1] = g2k 
        
        self.H[2,3] = g1k.conj()
        self.H[3,2] = g1k
        
        self.H[0,0], self.H[1,1], self.H[2,2], self.H[3,3] = self.M1 + f1k, self.M2-f1k, self.M3-f1k, self.M4+f1k
        
        self.H[0,3], self.H[1,2], self.H[2,1], self.H[3,0] = f2k, -f2k, -f2k, f2k
        
        #adding titlting term
        Htt = self.tt*(kx) * np.eye(4, dtype=complex)
        
        return np.array(self.H + Htt)
    
    