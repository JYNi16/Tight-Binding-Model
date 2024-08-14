# -*- coding: utf-8 -*-
"""
Python script to calculate berry curvature and Chern number of the Kane-mele band
Author: Jinyang Ni
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from BHZ_model import BHZ
import band_ini.config as cf
from Haldane_model import Honeycomb

dkp = 0.000001  #
numk = cf.numk # the density of k-points to calculate Berry Curvature
v = 0 
c = 1

#Ham = BHZ(-2.1)
Ham = Honeycomb(1, 0.1, 0.0, -0.0)

def H(k):
    return Ham.model(k)

def dHx(k):
    k2 = k - np.array([dkp,0])
    return (H(k) - H(k2))/dkp
def dHy(k):
    k2 = k - np.array([0,dkp])
    return (H(k) - H(k2))/dkp

#sorting the Eigenstates according to the Eigenvalues
def ewH(k):
    e,w=np.linalg.eigh(H(k))
    w0 = w[:, np.argsort(np.real(e))[0]]
    w1 = w[:, np.argsort(np.real(e))[1]]
    e = np.sort(np.real(e))
    return w0,w1,e[0],e[1]

#<v|dH/dk|c> v,c 
def vcdHx(v,k,c):
    dhc = np.dot(dHx(k), ewH(k)[c])
    vdhc = np.dot(ewH(k)[v].conj(),dhc)
    return vdhc

def vcdHy(v,k,c):
    dhc = np.dot(dHy(k), ewH(k)[c])
    vdhc = np.dot(ewH(k)[v].conj(),dhc)
    return vdhc  

def Omega(k):
    return 1.j *  (vcdHx(v,k,c) * vcdHy(c,k,v) - vcdHy(v,k,c) * vcdHx(c,k,v))/(ewH(k)[2]-ewH(k)[3])**2

def Chern_number():
    #xxx = np.linspace(-np.pi, np.pi, numk)
    #yyy = np.linspace(-np.pi, np.pi, numk)
    C = 0 
    sq3 = np.sqrt(3)
    dk = 2.0*np.pi/(numk-1)
    
    xx_h = cf.xx_h
    yy_h = cf.yy_h
    
    dky = (2*np.pi)/(numk-1)
    dkx = (4*np.pi*sq3/3)/(numk-1) 
    
    for i in range(numk):
        print("ith is:", i)
        for j in range(numk):
            k = np.array([xx_h[i], yy_h[i]])
            C+=np.real(Omega(k))
   
    print("Chern number is:", C/2/np.pi*dkx*dky)


def plot_berry():

    xx = np.linspace(-6,6,numk)
    yy = np.linspace(-6,6,numk)
    Z= np.zeros((numk,numk))
    X,Y = np.meshgrid(xx,yy)
    
    print("X.shape is:", X.shape)
    
    for i in range(numk):
        print("i th is:", i)
        for j in range(numk):
            k = np.array([xx[i],yy[j]])
            Z[i][j]=np.real(Omega(k))

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #        linewidth=0, antialiased=False)
    #ax.set_xlim3d(0.0, 200)
    #ax.set_ylim3d(0.0, 200)
    #ax.set_zlim3d(-0.90, 0.90)
    #plt.show()
    font = {'family': "Times New Roman", "weight":"normal", "size":20,}
    fig = plt.figure(figsize=(10,8))
    #plt.scatter(xx, yy, c=Z)
    plt.pcolormesh(X,Y,Z,  cmap='coolwarm', shading='gouraud')
    #C=plt.contour(X,Y,Z,10,colors='black',linewidths=0.1)
    #plt.clabel(C, inline=True,fontsize=10)
    plt.colorbar()
    plt.xlim(-6,6)
    plt.ylim(-6,6)
    plt.xlabel(r"$k_{x}$", font)
    plt.ylabel(r"$k_{y}$", font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("Berry_curvature.png", dpi=800)
    plt.show()


if __name__=="__main__":
    Chern_number()
    plot_berry()
