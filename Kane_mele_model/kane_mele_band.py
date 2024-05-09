# -*- coding: utf-8 -*-
"""
Spyder Editor

Python script to calculate the Kane-mele band
Author: Jinyang Ni
"""
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from k_sym_gen import *
from config import * 


#spin exchange parameters 
t1 = 1 ## NN hopping 
D = 0.1 ## NNN haldane hopping term

#define the Hamiltonian 
def H(k):
    H = np.zeros((2,2), dtype=complex)
    kx, ky = k[0], k[1]
    #print("kx is:", kx, "ky is:", ky)
    gk = np.exp(1.j*k.dot(a1)) + np.exp(1.j*k.dot(a2)) + np.exp(1.j*k.dot(a3))
    H[0,0] = 0
    H[0,1] = t1 * gk
    H[1,0] = t1 * gk.conj()
    H[1,1] = 0
    
    #add DM term 
    dk = np.sin(k.dot(d1)) + np.sin(k.dot(d2)) + np.sin(k.dot(d3))
    Hd = D*dk * sz
    
    return np.sort(np.linalg.eig(H + Hd)[0])

def band_post(k_syms):
    k_point_path, k_path, Node = k_path_sym_gen(k_syms)
    E_band = []
    
    for i in range(len(k_point_path)):
        E_values = np.array(list(map(H, k_point_path[i])))
        if (len(E_values.shape) < 2):
            E_band.append((np.reshape(E_values,[E_values.shape[0], -1])))
        else:
            E_band.append(E_values)
            
    return np.array(E_band)

#plot the band
def plot_band():    
    k_syms = [G, K, M, G]
    k_point_path, k_path, Node = k_path_sym_gen(k_syms)
    E_band = band_post(k_syms)
    shape = E_band.shape
    print("E_band.shape is:", shape)
    
    plt.figure(1, figsize=(6,6))
    if len(shape) < 2:
        eig = np.hstack(tuple(E_band))
        plt.plot(k_path, eig)
        #plt.xticks(Node,Node_label)
        plt.show()   
        return 
    
    for i in range(shape[-1]):
        eig_test = [] 
        for j in range(shape[0]):
            eig_test.append(E_band[j][:,i])
            print("eig_test.shape is:", len(eig_test))
        
        eig = np.hstack(tuple(eig_test))
        plt.plot(k_path, eig)
    
    k_sym_label = [r"$\Gamma$", "K", "M", r"$\Gamma$"]
    plt.xlim(0, k_path[-1])
    plt.xticks(Node, k_sym_label)
    
    plt.show()


if __name__=="__main__":
    plot_band()