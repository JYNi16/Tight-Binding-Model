# -*- coding: utf-8 -*-
"""
Created on 2025/1/8

A script to calculate the Wannier center for toy model 

@author: curry
"""

import numpy as np 
from BHZ_model import BHZ 
from Haldane_model import Honeycomb, stripe, Zigzag
from band_ini import config as cf
import matplotlib.pyplot as plt

#Ham = BHZ(-2.5)
Ham = Honeycomb(1, 0.1)
Ham_s = stripe()
Ham_z = Zigzag()

sq3 = np.sqrt(3)
Nband = 1

def Haldane_model(k):
    kx, ky = k
    
    H =np.zeros((2,2), dtype=complex)
    t1, D1, A = 1, -0.1, 0
    gk = np.exp(1.j*k.dot(cf.a1)) + np.exp(1.j*k.dot(cf.a2)) + np.exp(1.j*k.dot(cf.a3))
    H[0,1] = t1 * gk
    H[1,0] = t1 * gk.conj()
    
    #add NNN conj hopping term 
    dk = 2*np.sin(k.dot(cf.d1)) + np.sin(k.dot(cf.d2)) + np.sin(k.dot(cf.d3))
    Hd = D1*dk * cf.sz
    H2 = A * cf.sz 

def test_model(k):  
    kx, ky = k
    A=1
    B=0.3
    delta=1
    
    return 1*np.sin(kx)*cf.sx+1*np.sin(ky)*cf.sy+(delta-4*B*(np.sin(kx/2)**2)-4*B*np.sin(ky/2)**2)*cf.sz;

def ssh_2d(k):
    
    t1, t2, t3 = 0.2, 0.1, 1.3 
    
    tso1, tso2, tso3 = 0.5, 0.3, 0.0
    m1, m2 = 0, 0
    
    kx, ky = k
    Ho = t1*np.cos(kx)*np.kron(cf.s0,cf.s0)+t2*np.cos(ky)*np.kron(cf.s0, cf.s0)+t3*np.cos(ky/2)*np.kron(cf.sx, cf.s0)
    Hso1=tso1*np.sin(kx)*np.kron(cf.s0,cf.sz)
    Hso2 =tso2* np.sin(ky)*np.kron(cf.sz,cf.sz)
    Hso3 =tso3* np.cos(ky/2)*np.kron(cf.sy,cf.sz)

    M12 = m1*np.exp(1.j*ky/2)+m2*np.exp(-1.j*ky/2)
    M21 = m1*np.exp(-1.j*ky/2)+m2*np.exp(1.j*ky/2)
    M = np.array([[0, M12],[M21,0]])
    SSH = np.kron(M,cf.s0)


    hzmf=np.kron(np.eye(2),cf.sx)
    H=Ho+Hso1 + Hso2 + Hso3 + SSH + hzmf
    
    return np.array(H, dtype=complex)
    

def H(k):
    
    return Ham.model(k)

def ewH(k):
    e,w=np.linalg.eigh(H(k))
    
    s_idx = np.argsort(e)
    s_e = e[s_idx]
    s_w = w[:, s_idx]       
    return s_w

def Vmn(w1, w2, Ds): 
    for i in range(Nband):
        for j in range(Nband):
            Ds[i,j] = np.dot(w1[:,i].conj().T, w2[:,j])
    return Ds
            

def Wcc():
    #xx = np.linspace(-np.pi, np.pi, 101)
    #yy = np.linspace(-np.pi, np.pi, 101)
    xx = cf.xx_h
    yy = cf.yy_h
    wcc_kx = []
    for i in range(len(xx)):
        Ds = np.zeros((Nband, Nband), dtype=complex)
        vD = np.eye(Nband, dtype=complex)
        for j in range(len(yy)):
            VM = ewH(np.array([xx[i], yy[j]]))
            if j == len(yy)-1:
                j = 0
            else:
                j += 1
            VN = ewH(np.array([xx[i], yy[j]]))
            Ds = Vmn(VM, VN, Ds) 
            vD = np.dot(vD, Ds)
            #vD = vD*Ds
        
        #tranform the the eigenvalues to the complex type
        #e_arr = np.linalg.eigh(vD)[0].astype(complex)
        
        #print("eigen is:", np.linalg.eig(vD)[0])
        
        wcc_kx.append(np.imag(np.log(np.linalg.eig(vD)[0]))/(2*np.pi))
    
    return xx, np.array(wcc_kx)


def plot_wcc():
    xx, wcc_kx = Wcc()
    
    #print("wcc_ kx is:", wcc_kx)
    
    font = {'family': "Times New Roman", "weight":"normal", "size":24,}
    fig = plt.figure(figsize=(10,8))
    
    for i in range(Nband):
        plt.scatter(xx, wcc_kx[:,i], c= "none", s= 75, marker = "o", edgecolors="r")
    #plt.scatter(xx, yy, c=Z)
    #C=plt.contour(X,Y,Z,10,colors='black',linewidths=0.1)
    plt.yticks(fontproperties='Times New Roman', fontsize = 20)
    plt.xticks(fontproperties='Times New Roman', fontsize = 20)
    #plt.xlim(0,2*np.pi)
    plt.ylim(-0.55,0.55)
    plt.xlabel(r"$k_{x}$", font)
    plt.ylabel(r"$Wcc$", font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()
            


if __name__=="__main__":
    plot_wcc()