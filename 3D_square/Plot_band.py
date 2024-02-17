# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:29:19 2024

@author: Curry
"""

import numpy as np
import Hamiltonian as Ham
import matplotlib.pyplot as plt

#define the high-sym kpoints in the square lattice 
G = np.array([0,0,0])
R = np.array([np.pi, np.pi, np.pi])
M = np.array([np.pi, np.pi, 0])
X = np.array([0, np.pi, 0])
X2 = np.array([np.pi, 0, 0])

npoints = 50

Square_model = Ham.Square_3D()

def H(k):
    #ea = np.sort(np.real(np.linalg.eig(model.model_a(k))[0]))
    #eb = np.sort(np.real(np.linalg.eig(model.model_b(k))[0]))
    #e =  np.sort(np.linalg.eig(AFM_s.model(k))[0])
    e =  np.sort(np.linalg.eig(Square_model.model(k))[0])
    return e

def Dist(r1, r2):
    return np.linalg.norm(r1-r2)

def k_sym_path():
    kgr = np.linspace(G,R,npoints)
    krm = np.linspace(R,M,npoints)
    kmx = np.linspace(M,X,npoints)
    kxx2 = np.linspace(X,X2,npoints)
    kx2g = np.linspace(X2,G,npoints)
    
    k_point_path = [kgr, krm, kmx, kxx2, kx2g]
    
    lgr=Dist(G,R)
    lrm=Dist(R,M)
    lmx=Dist(M,X)
    lxx2=Dist(X,X2)
    lx2g=Dist(X2,G)

    lk = np.linspace(0, 1, npoints)
    xgr = lgr * lk 
    xrm = lrm * lk + xgr[-1]
    xmx = lmx * lk + xrm[-1]
    xxx2 = lxx2 * lk + xmx[-1]
    xx2g = lx2g * lk + xxx2[-1]
    
    kpath = np.concatenate((xgr, xrm, xmx, xxx2, xx2g), axis=0)
    
    Node = [0,  xgr[-1], xrm[-1], xmx[-1], xxx2[-1], xx2g[-1]]
    k_path = np.concatenate((xgr, xrm, xmx, xxx2, xx2g), axis=0)
    
    return k_point_path, k_path, Node

#define the Hamiltonian 

#    def FM_ana(self, k):
#        kx, ky = k[0], k[1]
#        gk = 1 + 4*np.cos(1.5*kx)*np.cos(sq3/2*ky) + 4*np.cos(sq3/2*ky)**2
#        e0 = 3*self.J1 + np.sqrt(self.FM.gk)
#        e1 = 3*self.J1 - np.sqrt(self.FM.gk)        
#        return [e0,e1]

#plot the band
    

def band_post():
    k_point_path, k_path, Node = k_sym_path()
    E_band = []
    
    for i in range(len(k_point_path)):
        E_values = np.array(list(map(H, k_point_path[i])))
        E_band.append(np.real(E_values))
    
    return E_band
    
def plot_band(): 
    
    font = {'family': "Times New Roman", "weight":"normal", "size":24,}
    
    k_point_path, k_path, Node = k_sym_path()
    E_band = np.array(band_post())
    shape = E_band.shape
    print("E_band.shape is:", shape)
    
    #np.save(save_path + "/E_band.npy", E_band)
        
    plt.figure(1, figsize=(10,8))
    if len(shape) <= 2:
        eig = np.hstack(tuple(E_band))
        plt.plot(k_path, eig, c="red", linewidth=4)
        #plt.xticks(Node,Node_label)
    
    else:
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
                
            
    
    k_sym_label =  [r"$\Gamma$", r"$R$", r"$M$", r"$X$", r"$X^{\prime}$", r"$\Gamma$"]
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