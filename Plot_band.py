# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:29:19 2024

@author: Curry
"""

import numpy as np

import matplotlib.pyplot as plt
import band_ini.config as cf
import band_ini.k_sym_gen as ksg

#import BHZ_model as Ham
import Haldane_model as Ham

Haldane = Ham.Honeycomb(1, 0.1)

def H(k):
    #ea = np.sort(np.real(np.linalg.eig(model.model_a(k))[0]))
    #eb = np.sort(np.real(np.linalg.eig(model.model_b(k))[0]))
    #e =  np.sort(np.linalg.eig(AFM_s.model(k))[0])
    e =  np.linalg.eigh(Haldane.model(k))[0]
    return e
    

def band_post(k_syms):
    k_point_path, k_path, Node = ksg.k_path_sym_gen(k_syms)
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
    k_syms = [cf.h_G, cf.h_M, cf.h_K, cf.h_G]
    k_point_path, k_path, Node = ksg.k_path_sym_gen(k_syms)
    E_band = band_post(k_syms)
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
        
    
    #k_sym_label =  [r"$X$", r"$\Gamma$", r"$M$"]
    k_sym_label =  [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"]
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