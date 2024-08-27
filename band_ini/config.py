import numpy as np
from numpy import pi

sq2 = np.sqrt(2)
sq3 = np.sqrt(3)

sq7 = np.sqrt(7)
sq21 = np.sqrt(21)
L1 = np.array([1.5*sq7, -0.5*sq21])
L2 = np.array([1.5*sq7, 0.5*sq21])

numk = 251

#High sym points of 1st BZ with 1D  
o_G = 0
o_X = pi
#X2 = np.array([np.pi, 0, 0])

#High sym points of 1st BZ with Square lattice 
s_G = np.array([0,0])
s_R = np.array([pi, pi])
s_X = np.array([0, pi])
s_M = np.array([pi, 0])
#X2 = np.array([np.pi, 0, 0])

#2*2 pauli matrices
s0 = np.array([[1,0], [0,1]])
sx = np.array([[0,1], [1,0]])
sy = np.array([[0,-1.j], [1.j, 0]])
sz = np.array([[1,0],[0,-1]])


#high-sym kpoints of 1st BZ in honeycomb lattice
#define high-sym kpoints in the honeycomb lattice
#G = np.array([0,0])
#K = np.array([sq3, 1])*(2*np.pi/(3))
#M = np.array([sq3, 0])*(2*np.pi/(3))
#K2 = np.array([sq3, -1])*(2*np.pi/(3))

G = np.array([0,0])
K = np.array([sq3, 1])*(2*np.pi/(3))
M = np.array([sq3, 0])*(2*np.pi/(3))
K2 = np.array([sq3, -1])*(2*np.pi/(3))

xx_h = np.linspace(0, (4*sq3*np.pi)/3, numk, endpoint=False)
yy_h = np.linspace((-2*np.pi)/(3), (4*np.pi)/(3), numk, endpoint=False)

#Kx and Ky in the fisrt brillouin zone of the Honeycomb lattice
#Left closed while right open !!!
#xx_h = np.linspace(0, (4*sq3*np.pi)/3, numk, endpoint=False)
#yy_h = np.linspace(0, (2*np.pi), numk, endpoint=False)


#NN link vectors honeycomb 
a1 = np.array(([-1/2, sq3/2])/sq3)
a2 = np.array(([-1/2, -sq3/2])/sq3)
a3 = np.array(([1,0])/sq3)

#NNN link vectors
d1 = np.array(([0, -sq3])/sq3)
d2 = np.array(([1.5, 0.5*sq3])/sq3)
d3 = np.array(([-1.5, 0.5*sq3])/sq3)

#NNNN link vectors
c1 = np.array(([-1.5, sq3/2])/sq3)
c2 = np.array(([1.5, sq3/2])/sq3)
c3 = np.array(([0, 1]))

#zigzag linking vectors 
#NN
az1 = np.array(([0,-1])/sq3)
az2 = np.array(([sq3/2, 1/2])/sq3)
az3 = np.array(([-sq3/2, 1/2])/sq3)

#NNN
dz1 = np.array(([sq3/2, 1.5])/sq3)
dz2 = np.array(([-sq3, 0])/sq3)
dz3 = np.array(([sq3/2, -1.5])/sq3)

#BZ high-sym in zigzag
Gz = np.array([0,0])
Kz = np.array([1, sq3])*(2*np.pi/(3))
Mz = np.array([1.5, sq3/2])*(2*np.pi/(3))
Xz = np.array([1.5, 0])*(2*np.pi/(3))
Yz = np.array([0, sq3/2])*(2*np.pi/(3))

#define the high-sym kpoints in twist_bilayer with 28 atoms 
G_t = np.array([0,0])
M_t = np.array([(2*sq7*np.pi)/21,0]) 
K_t = np.array([(2*sq7*np.pi)/21, (2*sq21*np.pi)/63]) 
K_t2 = np.array([(2*sq7*np.pi)/21, -(2*sq21*np.pi)/63]) 
