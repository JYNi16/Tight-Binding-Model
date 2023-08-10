# -*- coding: utf-8 -*-
"""
Spyder Editor

Python script to calculate the Kane-mele band
Author: Jinyang Ni
"""
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi

#NN vectors 
a1 = np.array([1/2, np.sqrt(3)/2])
a2 = np.array([1/2, -np.sqrt(3)/2])
a3 = np.array([-1,0])

#NNN vectors
d1 = np.array([0, -np.sqrt(3)])
d2 = np.array([1.5, 0.5*np.sqrt(3)])
d3 = np.array([-1.5, 0.5*np.sqrt(3)])

#define Pauli matrix 
sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,-1.j],[1.j, 0]])
s0 = np.array([[1,0],[0,1]])

#define high-sym kpoints
G = np.array([0,0])
K = np.array([np.sqrt(3), 1])*(2*pi/(3*np.sqrt(3)))
M = np.array([np.sqrt(3), 0])*(2*pi/(3*np.sqrt(3)))
npoints = 200

#spin exchange parameters 
J1 = 1
D = 0.1

def Dist(r1, r2):
    return np.linalg.norm(r1-r2)

def k_sym_path():
    kmg = np.linspace(M, G, npoints)
    kgk = np.linspace(G, K, npoints)
    kkm = np.linspace(K, M, npoints)
    
    k_point_path = [kmg, kgk, kkm]
    
    lmg = Dist(M,G)
    lgk = Dist(G,K)
    lkm = Dist(K,M)
    
    lk = np.linspace(0, 1, npoints)
    xmg = lmg * lk 
    xgk = lgk * lk + xmg[-1]
    xkm = lkm * lk + xgk[-1]
    
    Node = [0, xmg[-1], xgk[-1], xkm[-1]]
    k_path = np.concatenate((xmg, xgk, xkm), axis=0)
    
    return k_point_path, k_path, Node

#define the Hamiltonian 
def H_mag(k):
    H = np.zeros((2,2), dtype=complex)
    kx, ky = k[0], k[1]
    #print("kx is:", kx, "ky is:", ky)
    gk = np.exp(1.j*k.dot(a1)) + np.exp(1.j*k.dot(a2)) + np.exp(1.j*k.dot(a3))
    H[0,0] = 3*J1
    H[0,1] = J1 * gk
    H[1,0] = J1 * gk.conj()
    H[1,1] = 3*J1
    
    #add DM term 
    dk = np.sin(k.dot(d1)) + np.sin(k.dot(d2)) + np.sin(k.dot(d3))
    Hd = D*dk * sz
    
    return np.sort(np.linalg.eig(H + Hd)[0])

#plot the band
def plot_band(): 
    k_point_path, k_path, Node = k_sym_path()
    E_band = []
    
    for i in range(len(k_point_path)):
        E_values = np.array(list(map(H_mag, k_point_path[i])))
        print("E_values is:", E_values)
        E_band.append(E_values)
    
    shape = np.array(E_band).shape
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
    
    k_sym_label = ["M", r"$\Gamma$", "K", "M"]
    plt.xlim(0, k_path[-1])
    plt.xticks(Node, k_sym_label)
    
    plt.show()


if __name__=="__main__":
    plot_band()