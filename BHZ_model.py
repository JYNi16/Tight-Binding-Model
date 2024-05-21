# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:26:02 2024

@author: 26526
"""

import numpy as np 
import matplotlib.pyplot as plt 
from math import pi

from band_ini import config as cf

'''
BHZ model

'''
class BHZ():
    
    def __init__(self, u):
        self.H =np.zeros((2,2), dtype=complex)
        
        self.u = u 
        
    def model(self, k):
        kx, ky = k

        return np.array(self.H + np.sin(kx)*cf.sx + np.sin(ky)*cf.sy + (self.u + np.cos(kx) + np.cos(ky))*cf.sz) 