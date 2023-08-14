# -*- coding: utf-8 -*-
"""
Python script to calculate berry curvature of the Kane-mele band
Author: Jinyang Ni
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
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

#spin exchange parameters 
J1 = 1
D = 0.1

dkp = 0.000001  #用于求导的微元
numk = 201 # 用于计算贝里曲率的kx与ky的k点密度
v = 0 
c = 1

def H(k):
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
    
    return H+Hd

def dHx(k):
    k2 = k - np.array([dkp,0])
    return (H(k) - H(k2))/dkp
def dHy(k):
    k2 = k - np.array([0,dkp])
    return (H(k) - H(k2))/dkp

#按顺序匹配对应的本征值和本征矢
def ewH(k):
    e,w=np.linalg.eigh(H(k))
    w0 = w[:, np.argsort(np.real(e))[0]]
    w1 = w[:, np.argsort(np.real(e))[1]]
    e = np.sort(np.real(e))
    return w0,w1,e[0],e[1]

#<v|dH/dk|c> v,c 取0对应价带波函数，1对应导带波函数
def vcdHx(v,k,c):
    dhc = np.dot(dHx(k), ewH(k)[c])
    vdhc = np.dot(ewH(k)[v].conj(),dhc)
    return vdhc

def vcdHy(v,k,c):
    dhc = np.dot(dHy(k), ewH(k)[c])
    vdhc = np.dot(ewH(k)[v].conj(),dhc)
    return vdhc  

def Omega(k):
    print("this is ok")
    return 1.j *  (vcdHx(v,k,c) * vcdHy(c,k,v) - vcdHy(v,k,c) * vcdHx(c,k,v))/(ewH(k)[2]-ewH(k)[3])**2


def plot_berry():

    xx = np.linspace(-1.45*np.pi,1.45*np.pi,numk)
    yy = np.linspace(-1.45*np.pi,1.45*np.pi,numk)
    Z= np.zeros((numk,numk))
    X,Y = np.meshgrid(xx,yy)
    
    for i in range(numk):
        for j in range(numk):
            k = np.array([xx[i],yy[j]])
            print("Omega(k) is:",)
            Z[i][j]=np.real(Omega(k))

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #        linewidth=0, antialiased=False)
    #ax.set_xlim3d(0.0, num)
    #ax.set_ylim3d(0.0, num)
    #ax.set_zlim3d(-0.90, 0.90)
    #plt.show()
    
    fig = plt.figure(figsize=(5,5))
    plt.contourf(X,Y,Z,10,alpha=0.1,cmap=cm.RdYlGn)
    C=plt.contour(X,Y,Z,10,colors='black',linewidths=0.1)
    #plt.clabel(C, inline=True,fontsize=10)
    plt.show()


if __name__=="__main__":
    plot_berry()
