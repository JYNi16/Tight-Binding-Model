# -*- coding: utf-8 -*-
"""
Python script to calculate quantum geometry of bloch Haniltonians
Author: Jinyang Ni
"""
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np 
import matplotlib.pyplot as plt 
from math import pi
from BHZ_model import BHZ
import band_ini.config as cf
from Haldane_model import Honeycomb, stripe
from AFM_cdw import square_afm, Honey_Neel, bilayer_alter, square_alter
from matplotlib.colors import LinearSegmentedColormap
from Dirac_model import Dirac

dkp = 0.000001  #
numk = cf.numk # the density of k-points to calculate Berry Curvature
v = 0 
c = 1

class Quantum_geometry():
    
    def __init__(self): 
        
        #self.Ham = Ham_FM_SSH(J11=-1, J12=-1, J13=-1.5, D=0.0, J2=-0.05, A=-0.0, A0=-0.01)
        #self.Ham = Ham_FM() 
        self.Ham = stripe()
        #self.Ham = bilayer_alter()
        self.dim = self.Ham.model(np.array([0,0])).shape[0]
        self.h = 2 #2d_case:2; 3d_case:3

    def H(self,k): 
        return self.Ham.model(k)
    
    #only useful for 2 band model of honeycomb ferromagnets
    def dEky(self,k):
        return self.FM_ssh.dE_dky(k)
    
    #useful for multi-dimensions band model
    def dEky_ana(self,k): 
        w, e = self.ewH(k)
        k2 = k - np.array([0,dkp])
        w, e_dy = self.ewH(k2)
        dEy = []
        for v in range(self.dim):
            dEy.append((e[v] - e_dy[v])/dkp ) 
        return np.array(dEy)

    def dHx(self,k):
        if self.h < 3:
            k2 = k - np.array([dkp,0])
        else:
            k2 = k - np.array([dkp,0,0])
        return (self.H(k) - self.H(k2))/dkp

    def dHy(self,k):
        if self.h < 3:
            k2 = k - np.array([0,dkp])
        else:
            k2 = k - np.array([0,dkp,0])
        return (self.H(k) - self.H(k2))/dkp 

    #sorting the Eigenstates according to the Eigenvalues
    #def ewH(self,k):
    #    '''
    #    We should notice that np.linalg.eigh is used for Hermitian matrix
    #    np.linalg.eig is used for Non symmetric matrix
    #    '''   
    #    e,w=np.linalg.eig(self.H(k))
    #    w0 = w[:, np.argsort(np.real(e))[0]]
    #    w1 = w[:, np.argsort(np.real(e))[1]]
    #    e = np.sort(np.real(e))
    #    return w0,w1,e[0],e[1]
    
    def ewH(self, k):
        '''
        We should notice that np.linalg.eigh is used for Hermitian matrix
        np.linalg.eig is used for Non-Hermitian matrix
        '''  
        e,w=np.linalg.eigh(self.H(k))
        w_d = [] 
        for i in range(self.dim): 
            w_d.append(w[:, np.argsort(np.real(e))[i]])
        
        e_d = np.sort(np.real(e))
        
        return np.array(w_d), np.array(e_d)
    
    #<v|dH/dkx|c> v,c 
    #def vcdHx(self,v,k,c):
    #    dhc = np.dot(self.dHx(k), self.ewH(k)[c])
    #    vdhc = np.dot(self.ewH(k)[v].conj(),dhc)
    #    return vdhc

    #<v|dH/dky|c> v,c 
    #def vcdHy(self,v,k,c):
    #    dhc = np.dot(self.dHy(k), self.ewH(k)[c])
        #dhc = np.dot(FM_h.velocity(k)[1], ewH(k)[c])  
    #    vdhc = np.dot(self.ewH(k)[v].conj(),dhc)
    #    return vdhc 
    
    def vcdHx(self, v,k,c):
        dhc = np.dot(self.dHx(k), self.ewH(k)[0][c])
        #dhc = np.dot(self.FM.velocity(k)[0], self.ewH(k)[c])
        vdhc = np.dot(self.ewH(k)[0][v].conj(),dhc)
        return vdhc

    def vcdHy(self, v,k,c):
        dhc = np.dot(self.dHy(k), self.ewH(k)[0][c])
        #dhc = np.dot(self.FM.velocity(k)[1], self.ewH(k)[c])  
        vdhc = np.dot(self.ewH(k)[0][v].conj(),dhc)
        return vdhc

    #def Omega(self,k,v,c):
        
    #    return np.real(1.j*(self.vcdHx(v,k,c)*self.vcdHy(c,k,v) - self.vcdHy(v,k,c)*self.vcdHx(c,k,v))/(self.ewH(k)[2]-self.ewH(k)[3])**2)
    
    def Omega(self,k,v):
        '''
        Parameters
        ----------
        k : 2D array
            (kx, ky)
        v : int
            .

        Returns
        -------
        omega : TYPE
            DESCRIPTION.

        '''
        # the index of band
        band_idx = [] 
        for i in range(0, self.dim):
            if int(i) == v: 
                continue
            band_idx.append(int(i))
        
        Delta = 0.00001
        omega = 0 
        for c in band_idx: 
            # avoid the degenerate point
            #if abs(self.ewH(k)[1][v] - self.ewH(k)[1][c]) < 1e-8: 
            #    continue
            omega += 2*np.imag((self.vcdHx(v,k,c)*self.vcdHy(c,k,v))/(self.ewH(k)[1][v]-self.ewH(k)[1][c] + Delta)**2) 
            #omega += 2*(np.real((self.vcdHx(v,k,c)*self.vcdHy(c,k,v))/(self.ewH(k)[1][v]-self.ewH(k)[1][c] + Delta)**2))
        
        #print("ommega is:", omega)
        
        return omega


QE = Quantum_geometry()


def generate_input():
    xx = np.linspace(-6,6,numk)
    yy = np.linspace(-6,6, numk)
    #xx = np.linspace(-1,1, numk)
    #yy = np.linspace(-1,1, numk)
    Z= np.zeros((numk,numk))
    X,Y = np.meshgrid(xx,yy)
    
    return X,Y,Z
    

def cal_berry():

    X,Y,Z = generate_input()
    
    print("X.shape is:", X.shape)
    
    for i in range(numk):
        print("kx th is:", i)
        for j in range(numk):
            k = np.array([X[i][j],Y[i][j]])
            #print("k is:", k)
            #Z[i][j]=np.real(berry.mp_berry(k, 0, 1))
            Z[i][j]=np.real(QE.Omega(k, 1))
            #print(Z[i][j])
            #Z[i][j] = berry.dEky(k)[1]
            
    return X,Y,Z


def cal_metric():
    
    X,Y,Z = generate_input()
    
    print("X.shape is:", X.shape)
    
    for i in range(numk):
        print("kx th is:", i)
        for j in range(numk):
            k = np.array([X[i][j],Y[i][j]])
            #print("k is:", k)
            #Z[i][j]=np.real(berry.mp_berry(k, 0, 1))
            Z[i][j]=np.real(QE.Omega(k, 0))
            #print(Z[i][j])
            #Z[i][j] = berry.dEky(k)[1]
            
    return X,Y,Z
    
    

def plot_berry():
    
    X,Y,Z = cal_metric()
    print("calculation ok!!!")
    
    #save_path = "data_berry/FM/" + J_path_name
    #if os.path.exists(save_path):
    #    print("save path {} exist".format(save_path))
    #else:
    #    print("save path {} not exist".format(save_path))
    #    os.makedirs(save_path)
    #    print("now makedir the save_path")
    
    #np.save(save_path + "/X.npy", X)
    #np.save(save_path + "/Y.npy", Y)
    #np.save(save_path + "/Z.npy", Z)

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
    #        linewidth=0, antialiased=False)
    #ax.set_xlim3d(0.0, 200)
    #ax.set_ylim3d(0.0, 200)
    #ax.set_zlim3d(-0.90, 0.90)
    #plt.show()
    colors1 = [(0.0, 0.0, 1), (1,1,1), (1, 0.0, 0.0)]  # blue color to the red color
    cmap_name = 'custom_blue_red'
  
    cm1 = LinearSegmentedColormap.from_list(cmap_name, colors1, N=256)
    font = {'family': "Times New Roman", "weight":"normal", "size":24,}
    fig = plt.figure(figsize=(10,8))
    #plt.scatter(xx, yy, c=Z)
    #plt.tricontourf(X,Y,Z)
    plt.pcolormesh(X,Y,Z,  cmap=cm1, shading='gouraud')
    #C=plt.contour(X,Y,Z,10,colors='black',linewidths=0.1)
    #plt.clabel(C, inline=True,fontsize=10)
    plt.colorbar()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel(r"$k_{x}$", font)
    plt.ylabel(r"$k_{y}$", font)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.savefig("figure/Topological_Berry_curvature.png", dpi=800)
    plt.show()


if __name__=="__main__":
    #Chern_number()
    plot_berry()
