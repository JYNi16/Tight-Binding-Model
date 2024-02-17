# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:26:02 2024

@author: 26526
"""

import numpy as np 
import matplotlib.pyplot as plt 
from math import pi

'''
Hamiltonian of the monolayer honeycomb ferromagnets

'''

class Square_monolayer():
    
    def __init__(self, tAA=-1):
        self.H =np.zeros((2,2), dtype=complex)     
        self.tAA = tAA
        
    def model(self, k):
        kx, ky = k    
        return np.array(self.tA*(2*np.cos(kx) + 2*np.cos(ky)))



class Square_3D():
    
    def __init__(self, t1=-1, soc_r=-0.5, soc_i = -0):
        self.H =np.zeros((2,2), dtype=complex)
        
        self.t1 = t1
        
        #Rashba SOC 
        self.soc_r = soc_r
        
        #Intrinic SOC
        self.soc_i = soc_i
    
    def model(self, k):
        kx, ky, kz = k 
        
        self.H[0,0] = self.t1 * 2 * (np.cos(ky) + np.cos(kx) + np.cos(kz))
        self.H[1,1] = self.t1 * 2 * (np.cos(ky) + np.cos(kx) + np.cos(kz))
        
        self.H[0,1]  = 2 * self.soc_r * (1.j * np.sin(kx) + np.sin(ky))
        self.H[1,0]  = 2 * self.soc_r *(-1.j * np.sin(kx) + np.sin(ky))
        
        
        return self.H