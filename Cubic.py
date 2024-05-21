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

s0 = np.array([[1,0], [0,1]])
sx = np.array([[0,1], [1,0]])
sy = np.array([[0,-1.j], [1.j, 0]])
sz = np.array([[1,0],[0,-1]])
sq2 = np.sqrt(2)

class Square_3D():
    
    def __init__(self, t1=-1, t2 = -0.0, a1=0.0, b1=-0.0, a2=-0.0, b2=-0.0, a3=-0.0, b3=-0.0):
        self.H =np.zeros((2,2), dtype=complex)
        
        self.t1 = t1
        self.t2 = t2
        
        #SOC
        self.a1 = a1
        self.b1 = b1 
        self.a2 = a2
        self.b2 = b2
        self.a3 = a3
        self.b3 = b3

        self.A = 0.1
        
        self.H =np.zeros((2,2), dtype=complex)
        
    def model_nosoc(self, k):
        kx, ky, kz = k
        T2 = self.t2 * (2 * np.cos(sq2 * kx / 2) + 2 * np.cos(sq2 * ky / 2) + 2 * np.cos(sq2 * kz / 2))
        T1 = self.t1 * (2 * np.cos(kx / 2) + 2 * np.cos(ky / 2) + 2 * np.cos(kz / 2))
        self.H[0,1] = T1
        self.H[1,0] = T1
        self.H[0,0] = T2
        self.H[1,1] = T2


        return np.array(self.H + self.A*sz)
    
    def model(self, k):
        kx, ky, kz = k 
        
        #
        H_t = 2*self.t1*(np.cos(kx) + np.cos(ky) + np.cos(kz))*s0
        
        H_soc_1 = self.a1*(np.sin(ky)*sx -(np.sin(kx))*sy) + self.b1*(np.sin(ky)*sx + (np.sin(kx))*sy)
        H_soc_2 = self.a2*(np.sin(kz)*sx -(np.sin(kx))*sz) + self.b2*(np.sin(kz)*sx + (np.sin(kx))*sz)
        H_soc_3 = self.a3*(np.sin(kz)*sy -(np.sin(ky))*sz) + self.b2*(np.sin(kz)*sy + (np.sin(ky))*sz)
        
        return self.H + H_t + H_soc_1 + H_soc_2 + H_soc_3