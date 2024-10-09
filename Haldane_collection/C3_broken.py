"""
Standard Haldane model with broken C3 symmetry
Author: Peiyuan CUI
Date of creation: 2024-09-23
"""

import numpy as np
import matplotlib.pyplot as plt

import band_ini.k_sym_gen as ksg
from scipy.linalg import eigh, ishermitian


def C3_broken_hamiltonian(k):
    H = np.zeros((N_atom, N_atom), dtype=complex)
    H[0, 1] = (t1_all[0] * np.exp(-1j * k @ n1) +
               t1_all[1] * np.exp(-1j * k @ n2) +
               t1_all[2] * np.exp(-1j * k @ n3))
    H[1, 0] = np.conj(H[0, 1])

    H[0, 0] = (t2_all[0] * (e_i_phi * np.exp(-1j * k @ nn1) + e_mi_phi * np.exp(1j * k @ nn1)) +
               t2_all[1] * (e_i_phi * np.exp(-1j * k @ nn2) + e_mi_phi * np.exp(1j * k @ nn2)) +
               t2_all[2] * (e_i_phi * np.exp(-1j * k @ nn3) + e_mi_phi * np.exp(1j * k @ nn3))) + m
    H[1, 1] = (t2_all[0] * (e_mi_phi * np.exp(-1j * k @ nn1) + e_i_phi * np.exp(1j * k @ nn1)) +
               t2_all[1] * (e_mi_phi * np.exp(-1j * k @ nn2) + e_i_phi * np.exp(1j * k @ nn2)) +
               t2_all[2] * (e_mi_phi * np.exp(-1j * k @ nn3) + e_i_phi * np.exp(1j * k @ nn3))) - m
    return H


t_1 = 1
t_2 = 0.2
phi = np.pi / 2
m = 0.2

eta = 0.5

d = 2.46

a = np.array([[np.sqrt(3) / 2, -1 / 2, 0],
              [np.sqrt(3) / 2, 1 / 2, 0],
              [0, 0, 1 / d]]).T * d
a1 = a[:, 0]
a2 = a[:, 1]
a3 = a[:, 2]

V = a3 @ np.cross(a1, a2)
a1 = a1[:2]
a2 = a2[:2]

b = 2 * np.pi * np.linalg.inv(a).T

b1 = b[:2, 0]
b2 = b[:2, 1]

high_symmetry_points = np.array([[0, 0],  # Gamma
                                 [1 / 3, 2 / 3],  # K
                                 [1 / 2, 1 / 2],  # M
                                 [2 / 3, 1 / 3],  # K_prime
                                 [0, 0]], dtype=np.float64).T

R = np.array([b1, b2]).T

high_symmetry_points_cart = R @ high_symmetry_points
Gamma_cart = high_symmetry_points_cart[:, 0]
K_cart = high_symmetry_points_cart[:, 1]
M_cart = high_symmetry_points_cart[:, 2]
K_prime_cart = high_symmetry_points_cart[:, 3]

all_high_sym_points = [Gamma_cart, K_cart, M_cart, K_prime_cart, Gamma_cart]

k_point_path, k_path, Node = ksg.k_path_sym_gen(all_high_sym_points)
k_point_path = np.array(k_point_path).reshape((len(all_high_sym_points) - 1) * 100, 2)

N_atom = 2
all_eigv = np.zeros((N_atom, len(k_point_path)), dtype=np.double)

nearest_neighbor_vector = d / np.sqrt(3) * np.array([[1, 0],
                                                     [-1 / 2, np.sqrt(3) / 2],
                                                     [-1 / 2, -np.sqrt(3) / 2]]).T
n1 = nearest_neighbor_vector[:, 0]
n2 = nearest_neighbor_vector[:, 1]
n3 = nearest_neighbor_vector[:, 2]

t1_all = t_1 * np.array([1, eta, eta])

next_nearst_neighbor_vector = d * np.array([[0, 1],
                                            [-np.sqrt(3) / 2, -1 / 2],
                                            [np.sqrt(3) / 2, 1 / 2]]).T
nn1 = next_nearst_neighbor_vector[:, 0]
nn2 = next_nearst_neighbor_vector[:, 1]
nn3 = next_nearst_neighbor_vector[:, 2]

t2_all = t_2 * np.array([1, 1 / eta, 1 / eta])

e_i_phi = np.exp(1j * phi)
e_mi_phi = np.exp(-1j * phi)

for k_ind, k in enumerate(k_point_path[:]):
    H = C3_broken_hamiltonian(k)
    eigval, eigvec = eigh(H)
    all_eigv[:, k_ind] = eigval

plt.figure()
plt.plot(k_path, all_eigv.T)
plt.xticks(Node, ["G", "K", "M", "K'", "G"])
plt.title(f"Standard Haldane model with broken C3 symmetry \n eta = {eta}, phi = {phi}, m = {m}")
plt.show()
