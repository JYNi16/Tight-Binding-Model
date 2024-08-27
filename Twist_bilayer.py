# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:56:29 2024

@author: 26526
"""


import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from band_ini import config as cf

sq7 = np.sqrt(7)
sq21 = np.sqrt(21)
L1 = np.array([1.5*sq7, -0.5*sq21])
L2 = np.array([1.5*sq7, 0.5*sq21])

class twist_FM_1():
    def __init__(self, J1=3):
        self.H =np.zeros((7,7), dtype=complex)
        self.J1 = J1
        self.D = -0.1
        self.Jc = -0.2
        self.A0 = 0.5
        
        #tilting term 
        self.tt = -0.5
        
        #Haldane flux term 
        self.fk = self.D*1.j
        self.fk_c = -self.D*1.j
        
    def H11_11(self, k):
        
        #H11_11 1st
        H =np.zeros((7,7), dtype=complex)
        H[0,1] = self.J1
        H[1,0] = self.J1 
        
        H[0,5] = self.J1*np.exp(-1.j*k.dot(L1))
        H[5,0] = H[0,5].conj()
        
        H[1,2] = self.J1
        H[2,1] = self.J1 
        
        H[2,2] = -self.Jc
        
        H[2,3] = self.J1
        H[3,2] = self.J1
        
        H[3,4] = self.J1
        H[4,3] = self.J1 
        
        H[4,5] = self.J1
        H[5,4] = self.J1
        
        H[5,6] = self.J1
        H[6,5] = self.J1
        
        #2nd 
        H[0,2] = self.fk
        H[2,0] = H[0,2].conj() 
        
        H[0,4] = self.fk_c*np.exp(-1.j*k.dot(L1))
        H[4,0] = H[0,4].conj()
        
        H[0,6] = self.fk*np.exp(-1.j*k.dot(L1))
        H[6,0] = H[0,6].conj()
        
        H[1,3] = self.fk
        H[3,1] = H[1,3].conj()
        
        H[1,5] = self.fk*np.exp(-1.j*k.dot(L1))
        H[5,1] = H[1,5].conj()
        
        H[2,4] = self.fk_c 
        H[4,2] = H[2,4].conj()
        
        H[2,6] = self.fk
        H[6,2] = H[6,2].conj()
        
        H[3,5] = self.fk_c 
        H[5,3] = H[3,5].conj()
        
        H[4,6] = self.fk_c
        H[6,4] = H[4,6].conj() 

        return H
    
    def H11_12(self, k):
        
        #H11_12 1st
        H =np.zeros((7,7), dtype=complex)
        H[0,2] = self.J1*np.exp(-1.j*k.dot(L2))
        
        H[1,5] = self.J1*np.exp(-1.j*k.dot(L1))
       
        H[2,0] = self.J1
        
        H[3,3] = self.J1*np.exp(-1.j*k.dot(L2))
        
        H[4,6] = self.J1*np.exp(-1.j*k.dot(L2))
        
        H[6,0] = self.J1
        H[6,4] = self.J1
        
        #2nd
        H[0,1] = self.fk * np.exp(-1.j*k.dot(L2))
        H[0,3] = self.fk_c * np.exp(-1.j*k.dot(L2))
        H[0,5] = self.fk_c * np.exp(-1.j*k.dot(L1))
        
        H[1,0] = self.fk_c  
        H[1,2] = self.fk_c * np.exp(-1.j*k.dot(L2)) 
        H[1,4] = self.fk_c * np.exp(-1.j*k.dot(L1))
        H[1,6] = self.fk * np.exp(-1.j*k.dot(L1))
        
        H[2,1] = self.fk_c
        H[2,3] = self.fk * np.exp(-1.j*k.dot(L2))
        H[2,5] = self.fk * np.exp(-1.j*k.dot(L1))
        
        H[3,0] = self.fk
        H[3,2] = self.fk * np.exp(-1.j*k.dot(L2))
        H[3,4] = self.fk_c * np.exp(-1.j*k.dot(L2))
        H[3,6] = self.fk * np.exp(-1.j*k.dot(L2))
        
        H[4,1] = self.fk_c * np.exp(-1.j*k.dot(L2-L1))
        H[4,3] = self.fk_c * np.exp(-1.j*k.dot(L2))
        H[4,5] = self.fk * np.exp(-1.j*k.dot(L2))
        
        H[5,0] = self.fk_c
        H[5,2] = self.fk * np.exp(-1.j*k.dot(L2-L1))
        H[5,4] = self.fk 
        H[5,6] = self.fk_c * np.exp(-1.j*k.dot(L2))
        
        H[6,1] = self.fk
        H[6,3] = self.fk_c
        H[6,5] = self.fk
         
        return H
    
    def H11_21(self, k):
        
        return np.conjugate(self.H11_12(k)).T
    
    def H11_22(self, k):
        H =np.zeros((7,7), dtype=complex)
        H[0,1] = self.J1
        H[1,0] = self.J1 
        
        H[1,2] = self.J1 
        H[2,1] = self.J1
        
        H[1,6] = self.J1*np.exp(-1.j*k.dot(L1))
        H[6,1] = self.J1*np.exp(1.j*k.dot(L1))
        
        H[2,3] = self.J1 
        H[3,2] = self.J1 
        
        H[3,4] = self.J1
        H[4,3] = self.J1
        
        H[4,4] = -self.Jc 
        
        H[4,5] = self.J1
        H[5,4] = self.J1
        
        H[5,6] = self.J1
        H[6,5] = self.J1
        
        #2nd 
        H[0,2] = self.fk
        H[2,0] = H[0,2].conj() 
        
        H[0,4] = self.fk_c
        H[4,0] = H[0,4].conj()
        
        H[0,6] = self.fk_c*np.exp(-1.j*k.dot(L1))
        H[6,0] = H[0,6].conj()
        
        H[1,3] = self.fk
        H[3,1] = H[1,3].conj()
        
        H[1,5] = self.fk_c*np.exp(-1.j*k.dot(L1))
        H[5,1] = H[1,5].conj()
        
        H[2,4] = self.fk 
        H[4,2] = H[2,4].conj()
        
        H[2,6] = self.fk*np.exp(-1.j*k.dot(L1))
        H[6,2] = H[2,6].conj()
        
        H[3,5] = self.fk_c 
        H[5,3] = H[3,5].conj()
        
        H[4,6] = self.fk_c
        H[6,4] = H[4,6].conj() 
        
        
        return H
    
    def H12_11(self, k):
        H =np.zeros((7,7), dtype=complex)
        
        H[2,2] = self.Jc
        
        return H 
    
    def H12_22(self, k):
        H =np.zeros((7,7), dtype=complex)
        
        H[4,4] = self.Jc
        
        return H 
    
    def H22_11(self, k):
        
        #H11_11
        H =np.zeros((7,7), dtype=complex)
        H[0,1] = self.J1
        H[1,0] = self.J1 
        
        H[0,5] = self.J1*np.exp(-1.j*k.dot(L2))
        H[5,0] = H[0,5].conj()
        
        H[1,2] = self.J1
        H[2,1] = self.J1 
        
        H[2,2] = -self.Jc
        
        H[2,3] = self.J1
        H[3,2] = self.J1
        
        H[3,4] = self.J1
        H[4,3] = self.J1 
        
        H[4,5] = self.J1
        H[5,4] = self.J1
        
        H[5,6] = self.J1
        H[6,5] = self.J1
        
        #2nd 
        H[0,2] = self.fk
        H[2,0] = H[0,2].conj() 
        
        H[0,4] = self.fk_c*np.exp(-1.j*k.dot(L2))
        H[4,0] = H[0,4].conj()
        
        H[0,6] = self.fk*np.exp(-1.j*k.dot(L2))
        H[6,0] = H[0,6].conj()
        
        H[1,3] = self.fk
        H[3,1] = H[1,3].conj()
        
        H[1,5] = self.fk*np.exp(-1.j*k.dot(L2))
        H[5,1] = H[1,5].conj()
        
        H[2,4] = self.fk_c 
        H[4,2] = H[2,4].conj()
        
        H[2,6] = self.fk
        H[6,2] = H[2,6].conj()
        
        H[3,5] = self.fk_c 
        H[5,3] = H[3,5].conj()
        
        H[4,6] = self.fk_c
        H[6,4] = H[4,6].conj() 
         
        return H
    
    def H22_12(self, k):
        
        #H11_12
        H =np.zeros((7,7), dtype=complex)
        H[0,2] = self.J1*np.exp(-1.j*k.dot(L1))
        
        H[1,5] = self.J1*np.exp(-1.j*k.dot(L2))
       
        H[2,0] = self.J1
        
        H[3,3] = self.J1*np.exp(-1.j*k.dot(L1))
        
        H[4,6] = self.J1*np.exp(-1.j*k.dot(L1))
        
        H[6,0] = self.J1
        H[6,4] = self.J1
        
        #2nd
        H[0,1] = self.fk * np.exp(-1.j*k.dot(L1))
        H[0,3] = self.fk_c * np.exp(-1.j*k.dot(L1))
        H[0,5] = self.fk_c * np.exp(-1.j*k.dot(L2))
        
        H[1,0] = self.fk_c  
        H[1,2] = self.fk_c * np.exp(-1.j*k.dot(L1)) 
        H[1,4] = self.fk_c * np.exp(-1.j*k.dot(L2))
        H[1,6] = self.fk * np.exp(-1.j*k.dot(L2))
        
        H[2,1] = self.fk_c
        H[2,3] = self.fk * np.exp(-1.j*k.dot(L1))
        H[2,5] = self.fk * np.exp(-1.j*k.dot(L2))
        
        H[3,0] = self.fk
        H[3,2] = self.fk * np.exp(-1.j*k.dot(L1))
        H[3,4] = self.fk_c * np.exp(-1.j*k.dot(L1))
        H[3,6] = self.fk * np.exp(-1.j*k.dot(L1))
        
        H[4,1] = self.fk_c * np.exp(-1.j*k.dot(L2-L1))
        H[4,3] = self.fk_c * np.exp(-1.j*k.dot(L1))
        H[4,5] = self.fk * np.exp(-1.j*k.dot(L1))
        
        H[5,0] = self.fk_c
        H[5,2] = self.fk * np.exp(-1.j*k.dot(L2-L1))
        H[5,4] = self.fk 
        H[5,6] = self.fk_c * np.exp(-1.j*k.dot(L1))
        
        H[6,1] = self.fk
        H[6,3] = self.fk_c
        H[6,5] = self.fk
         
        return H
    
    def H22_21(self, k):
        
        return np.conjugate(self.H22_12(k)).T
    
    def H22_22(self, k):
        H =np.zeros((7,7), dtype=complex)
        H[0,1] = self.J1
        H[1,0] = self.J1 
        
        H[1,2] = self.J1 
        H[2,1] = self.J1
        
        H[1,6] = self.J1*np.exp(-1.j*k.dot(L2))
        H[6,1] = self.J1*np.exp(1.j*k.dot(L2))
        
        H[2,3] = self.J1 
        H[3,2] = self.J1 
        
        H[3,4] = self.J1 
        H[4,3] = self.J1
        
        H[4,4] = -self.Jc 
        
        H[4,5] = self.J1
        H[5,4] = self.J1
        
        H[5,6] = self.J1
        H[6,5] = self.J1
        
        #2nd 
        H[0,2] = self.fk
        H[2,0] = H[0,2].conj() 
        
        H[0,4] = self.fk_c
        H[4,0] = H[0,4].conj()
        
        H[0,6] = self.fk_c*np.exp(-1.j*k.dot(L2))
        H[6,0] = H[0,6].conj()
        
        H[1,3] = self.fk
        H[3,1] = H[1,3].conj()
        
        H[1,5] = self.fk_c*np.exp(-1.j*k.dot(L2))
        H[5,1] = H[1,5].conj()
        
        H[2,4] = self.fk 
        H[4,2] = H[2,4].conj()
        
        H[2,6] = self.fk*np.exp(-1.j*k.dot(L2))
        H[6,2] = H[2,6].conj()
        
        H[3,5] = self.fk_c 
        H[5,3] = H[3,5].conj()
        
        H[4,6] = self.fk_c
        H[6,4] = H[4,6].conj() 
        
        return H
    
    def model(self, k):
        
        kx, ky = k
        
        H1_up = np.concatenate((self.H11_11(k), self.H11_12(k)), axis=1)
        H1_dn = np.concatenate((self.H11_21(k), self.H11_22(k)), axis=1)
        H1 = np.concatenate((H1_up, H1_dn), axis=0)
        
        H2_up = np.concatenate((self.H22_11(k), self.H22_12(k)), axis=1)
        H2_dn = np.concatenate((self.H22_21(k), self.H22_22(k)), axis=1)
        H2 = np.concatenate((H2_up, H2_dn), axis=0)
        
        H_jc_1 = np.concatenate((self.H12_11(k), np.zeros((7,7),dtype=complex)), axis=1)
        H_jc_2 = np.concatenate((np.zeros((7,7),dtype=complex), self.H12_22(k)), axis=1)
        H_jc1 = np.concatenate((H_jc_1, H_jc_2), axis=0)
        H_jc2 = np.conjugate(H_jc1).T
        
        H_up = np.concatenate((H1, H_jc1), axis=1)
        H_dn = np.concatenate((H_jc2, H2), axis=1)
        
        H = np.concatenate((H_up, H_dn), axis=0)
        
        Htt = (self.A0 + self.tt * (kx + ky)) * np.eye(28, dtype=complex)
        
        return np.array(H + Htt )

if __name__ == "__main__":
    FM = twist_FM_1()
    
    H = FM.model(np.array([1,1]))
    
    print(H.shape)
        
        