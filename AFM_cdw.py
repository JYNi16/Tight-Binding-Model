# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 17:30:14 2024

@author: 26526
"""

import numpy as np 
import matplotlib.pyplot as plt 
from math import pi

from band_ini import config as cf

class square_afm():
    def __init__(self):
        self.t1 = 1
        self.l = 0.3
        self.H = np.zeros((4,4), dtype=complex) 
        
        #spin term 
        self.Sz = -2
        self.Sy = 0
        self.Sx = 0
        
        #cdw potential
        self.M = 0.4
    
    def model(self, k):
        kx, ky = k
        fk = -self.t1 *(np.cos(kx) + np.cos(ky))
        
        S = self.Sx * cf.sx + self.Sy * cf.sy + self.Sz *cf.sz 
        
        M = self.M * cf.s0
        
        H_right = fk * cf.s0 + self.l * (np.sin(kx) * cf.sy + np.sin(ky) * cf.sx)
        H_left = fk * cf.s0 + self.l * (np.sin(kx) * cf.sy + np.sin(ky) * cf.sx)
        
        H1 = np.concatenate(((S+M), H_right), axis=1)
        H2 = np.concatenate((H_left, (-S-M)), axis=1)
        
        H = np.concatenate((H1, H2), axis=0)
        
        return np.array(H)

class Honey_Neel():
    
    def __init__(self):
        self.t1 = 1
        self.l = 0.3
        self.H = np.zeros((4,4), dtype=complex) 
        
        self.M = 0.2
        
        #spin moments
        #spin term 
        self.Sz = 1
        self.Sy = 0
        self.Sx = 0
    
    def model(self, k):
        kx, ky = k 
        
        S = self.Sx * cf.sx + self.Sy * cf.sy + self.Sz *cf.sz 
        
        M = self.M * cf.s0
        
        r_a3 = -cf.sy*(cf.a3[0])*np.exp(1.j*k.dot(cf.a3)) 
        r_a1 = (cf.sx*cf.a1[0] - cf.a1[1]*cf.sy)*np.exp(1.j*k.dot(cf.a1))
        r_a2 = (cf.sx*cf.a2[0] - cf.a2[1]*cf.sy)*np.exp(1.j*k.dot(cf.a2)) 
        
        H_rashba = self.l * (r_a1 + r_a2 + r_a3)
        
        H_t0 = -cf.s0*(self.t1*(np.exp(1.j*k.dot(cf.a1)) + np.exp(1.j*k.dot(cf.a2)) + np.exp(1.j*k.dot(cf.a3))))
        
        H_right = H_t0 + H_rashba 
        
        H_left = H_t0.conj() + H_rashba
        
        H1 = np.concatenate(((S+M), H_right), axis=1)
        H2 = np.concatenate((H_left, (-S-M)), axis=1)
        
        H = np.concatenate((H1, H2), axis=0)
        
        return np.array(H)
    
    
    
        