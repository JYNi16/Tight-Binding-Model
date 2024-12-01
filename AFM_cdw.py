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
        self.l = 0.4
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
    
    def __init__(self, t1):
        self.t1 = t1
        self.l = 0.1
        self.H = np.zeros((4,4), dtype=complex) 
        
        self.M = -0.0
        
        #spin moments
        self.S = 1
    
    def model(self, k):
        kx, ky = k 
        
        self.H[0,0], self.H[1,1] = self.M + self.S, self.M - self.S 
        self.H[2,2], self.H[3,3] = -self.M - self.S, -self.M + self.S
        
        gk = np.exp(1.j*k.dot(cf.a1)) + np.exp(1.j*k.dot(cf.a2)) + np.exp(1.j*k.dot(cf.a3))
        self.H[0,2] = self.t1 * gk
        self.H[2,0] = self.t1 * gk.conj()
    
        #add NNN conj hopping term 
        dk = 2*np.sin(k.dot(cf.d1)) + np.sin(k.dot(cf.d2)) + np.sin(k.dot(cf.d3))
        Hd = self.D1*dk * cf.sz
        jk = self.J2*(2*(np.cos(k.dot(cf.d1)) + np.cos(k.dot(cf.d2)) + np.cos(k.dot(cf.d3))) - 6)
        H2 = jk * cf.sz + self.A * cf.sz + self.A0 * cf.s0 + self.tt*(kx + ky) * cf.s0
        return np.array(self.H + Hd + H2)
    
    
    
        