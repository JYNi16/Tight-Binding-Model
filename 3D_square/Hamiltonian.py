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
    
    def __init__(self, e0=0, tab=-1, tc = -1,  soc_r=-0.5):
        self.H =np.zeros((2,2), dtype=complex)
        
        self.tab = tab
        self.tc = tc
        self.e0 = e0
        
        #Rashba SOC 
        self.soc_r = soc_r
        
    
    def model(self, k):
        kx, ky, kz = k 
        
        self.H[0,0] = self.tab*2*(np.cos(ky) + np.cos(kx)) + self.tc*2*(np.cos(kz)) 
        self.H[1,1] = self.tab*2*(np.cos(ky) + np.cos(kx)) + self.tc*2*(np.cos(kz))
        
        #Rashba SOC
        L_r = 2*self.soc_r*(1.j*np.sin(kx) + np.sin(ky))
        
        self.H[0,1] = L_r
        self.H[1,0] = L_r.conjugate()
        
        return self.H