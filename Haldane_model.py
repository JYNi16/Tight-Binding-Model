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
t1 = 1 ## NN hopping 
D = 0.1 ## NNN haldane hopping term

#define the Hamiltonian 

class Honeycomb():
    
    def __init__(self, t1, D1):
        self.H =np.zeros((2,2), dtype=complex)
        self.t1 = t1 
        self.D1 = D1
    
    def model(self, k):
        gk = np.exp(1.j*k.dot(cf.a1)) + np.exp(1.j*k.dot(cf.a2)) + np.exp(1.j*k.dot(cf.a3))
        self.H[0,1] = self.t1 * gk
        self.H[1,0] = self.t1 * gk.conj()
    
        #add NNN conj hopping term 
        dk = np.sin(k.dot(cf.d1)) + np.sin(k.dot(cf.d2)) + np.sin(k.dot(cf.d3))
        Hd = self.D1*dk * cf.sz
        
        return np.array(self.H + Hd)