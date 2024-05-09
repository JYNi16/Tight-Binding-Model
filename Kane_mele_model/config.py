import numpy as np
from numpy import pi

#define high-sym kpoints
G = np.array([0,0])
K = np.array([np.sqrt(3), 1])*(2*pi/(3*np.sqrt(3)))
M = np.array([np.sqrt(3), 0])*(2*pi/(3*np.sqrt(3)))
npoints = 200

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

font = {'family': "Times New Roman", "weight":"normal", "size":24,}