# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:29:19 2024

@author: Curry
"""

import numpy as np
import Hamiltonian as Ham
import matplotlib.pyplot as plt
from k_sym_gen import *

G = np.array([0,0,0])
R = np.array([np.pi, np.pi, np.pi])
X = np.array([0, np.pi, 0])
M = np.array([np.pi, np.pi, 0])
#X2 = np.array([np.pi, 0, 0])

npoints = 50

Square_model = Ham.Square_3D()

def H(k):
    #ea = np.sort(np.real(np.linalg.eig(model.model_a(k))[0]))
    #eb = np.sort(np.real(np.linalg.eig(model.model_b(k))[0]))
    #e =  np.sort(np.linalg.eig(AFM_s.model(k))[0])
    e =  np.linalg.eigh(Square_model.model_nosoc(k))[0]
    return e

#define the Hamiltonian 

#    def FM_ana(self, k):
#        kx, ky = k[0], k[1]
#        gk = 1 + 4*np.cos(1.5*kx)*np.cos(sq3/2*ky) + 4*np.cos(sq3/2*ky)**2
#        e0 = 3*self.J1 + np.sqrt(self.FM.gk)
#        e1 = 3*self.J1 - np.sqrt(self.FM.gk)        
#        return [e0,e1]

#plot the band
    

def band_post():
    k_syms = [G, R, X, M]
    k_point_path, k_path, Node = k_path_sym_gen(k_syms)
    E_band = []
    
    for i in range(len(k_point_path)):
        E_values = np.array(list(map(H, k_point_path[i])))
        if (len(E_values.shape) < 2):
            E_band.append((np.reshape(E_values,[E_values.shape[0], -1])))
        else:
            E_band.append(E_values)
            
    return np.array(E_band)
    
def plot_band(): 
    
    font = {'family': "Times New Roman", "weight":"normal", "size":24,}
    k_syms = [G, R, X, M]
    k_point_path, k_path, Node = k_path_sym_gen(k_syms)
    E_band = band_post()
    shape = E_band.shape
    print("E_band.shape is:", shape)
    
    #np.save(save_path + "/E_band.npy", E_band)
        
    plt.figure(1, figsize=(10,8))
    
    for i in range(shape[-1]):
        eig_test = [] 
        for j in range(shape[0]):
            eig_test.append(E_band[j][:,i])
            print("eig_test.shape is:", len(eig_test))
            
        eig = np.hstack(tuple(eig_test))
        if i == 0:
            plt.plot(k_path, eig, c="red", linewidth=2)
        else:
            plt.plot(k_path, eig, c="seagreen", linewidth=2)
        
    
    k_sym_label =  [r"$\Gamma$", r"$R$", r"$X$", r"$M$"]
    plt.xlim(0, k_path[-1])
    #plt.ylim(0, 1.2)
    plt.xticks(Node, k_sym_label, fontproperties = "Times New Roman", fontsize=24) 
    plt.xlabel("$K$-points", font)
    plt.ylabel("Energy($meV$)", font)
    #font_txt = {'style': "normal", "weight":"normal", "size":20, 'family': "Times New Roman"}
    plt.xticks(fontproperties = "Times New Roman", fontsize=24)
    plt.yticks(fontproperties = "Times New Roman", fontsize=24)
    
    plt.show()


if __name__=="__main__":
    plot_band()