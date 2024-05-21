import numpy as np
from numpy import pi

#High sym points of 1st BZ with Square lattice 
s_G = np.array([0,0])
s_R = np.array([pi, pi])
s_X = np.array([0, pi])
s_M = np.array([0, pi])
#X2 = np.array([np.pi, 0, 0])

#2*2 pauli matrices
s0 = np.array([[1,0], [0,1]])
sx = np.array([[0,1], [1,0]])
sy = np.array([[0,-1.j], [1.j, 0]])
sz = np.array([[1,0],[0,-1]])
sq2 = np.sqrt(2)

#high-sym kpoints of 1st BZ in honeycomb lattice
h_G = np.array([0,0])
h_K = np.array([np.sqrt(3), 1])*(2*pi/(3*np.sqrt(3)))
h_M = np.array([np.sqrt(3), 0])*(2*pi/(3*np.sqrt(3)))

#NN vectors 
a1 = np.array([1/2, np.sqrt(3)/2])
a2 = np.array([1/2, -np.sqrt(3)/2])
a3 = np.array([-1,0])

#NNN vectors
d1 = np.array([0, -np.sqrt(3)])
d2 = np.array([1.5, 0.5*np.sqrt(3)])
d3 = np.array([-1.5, 0.5*np.sqrt(3)])
