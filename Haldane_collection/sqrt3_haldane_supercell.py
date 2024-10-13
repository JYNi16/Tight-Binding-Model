import numpy as np
import matplotlib.pyplot as plt
import band_ini.k_sym_gen as ksg
from scipy.linalg import eigh, ishermitian
import sys

# W = 50  #width of the ribbon
#
# t_1 = 1
# t_2 = 0.1
# phi = np.pi / 4

# W = 2 # width of the ribbon
t_1 = 1
t_2 = 0.0
# t_2 = 0.1842
phi = np.pi / 2

# tilting = 0.6
tilting = 0.3

# onsite = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3]) * 3
# onsite = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) * 3
# onsite = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1]) * 4
onsite = [0, 0, 0, 0, 0, 0]

e_i_phi = np.exp(1j * phi)
e_mi_phi = np.exp(-1j * phi)

d = 2.46 * np.sqrt(3)

# a = np.array([[np.sqrt(3) / 2, -1 / 2, 0],
#               [np.sqrt(3) / 2, 1 / 2, 0],
#               [0, 0, 1 / d]]).T * d

a = np.array([[1, 0, 0],
              [1 / 2, np.sqrt(3) / 2, 0],
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

# High symmetry points in fractional coordinates
Gamma = np.array([0, 0])
M = np.array([0.5, 0.5])
K = np.array([2 / 3, 1 / 3])
K_prime = np.array([1 / 3, 2 / 3])

R = np.array([b1, b2]).T
Gamma_cart = R @ Gamma
M_cart = R @ M
K_cart = R @ K
K_prime_cart = R @ K_prime

# -30 degree
# Rotation=np.array([[np.cos(-np.pi/6),-np.sin(-np.pi/6)],[np.sin(-np.pi/6),np.cos(-np.pi/6)]])
#
Gamma_cart = np.array([0.000000, 0.000000])
K_cart = np.array([1.474634, 0.851380])
M_cart = np.array([1.474634, 0.000000])
K_prime_cart = np.array([1.474634, -0.851380])

all_high_sym_points = [Gamma_cart, K_cart, M_cart, K_prime_cart, Gamma_cart]
# all_high_sym_points = [Gamma_cart, K_cart, M_cart, K_prime_cart]

k_point_path, k_path, Node = ksg.k_path_sym_gen(all_high_sym_points)
k_point_path = np.array(k_point_path).reshape((len(all_high_sym_points) - 1) * 100, 2)
N_atom = len(onsite)
all_eigv = np.zeros((N_atom, len(k_point_path)), dtype=np.double)

for k_ind, k in enumerate(k_point_path[:]):
    # print(k)
    e_ika1 = np.exp(1j * k @ a1)
    e_mi_ka1 = np.exp(-1j * k @ a1)
    e_ika2 = np.exp(1j * k @ a2)
    e_mi_ka2 = np.exp(-1j * k @ a2)
    print("k", k_ind, k)
    # print(f"ka1, mi_ka1, ka2, mi_ka2, {k @ a1}, {k @ a2}, {k @ a1}, {k @ a2}")
    # print(e_ika1, e_mi_ka1, e_ika2, e_mi_ka2)

    H = np.zeros((N_atom, N_atom), dtype=complex)
    # on-site energy
    H[0, 0] = onsite[0]
    H[1, 1] = onsite[1]
    H[2, 2] = onsite[2]
    H[3, 3] = onsite[3]
    H[4, 4] = onsite[4]
    H[5, 5] = onsite[5]

    H += np.eye(N_atom) * tilting * (k[0] + k[1])

    # Near neighbor hopping
    H[0, 1] = t_1  # 2 to 1, intra-cell
    H[0, 3] = t_1  # 4 to 1, intra-cell
    H[0, 5] = t_1 * e_ika2  # 6 to 1, a2 side neighbor

    H[1, 0] = t_1  # 1 to 2, intra-cell
    H[1, 2] = t_1  # 3 to 2, intra-cell
    H[1, 4] = t_1 * e_mi_ka1 * e_ika2  # 5 to 2, a1-a2 side neighbor

    H[2, 1] = t_1  # 2 to 3, intra-cell
    H[2, 3] = t_1 * e_mi_ka1  # 4 to 3, a1 side neighbor
    H[2, 5] = t_1  # 6 to 3, intra-cell

    H[3, 0] = t_1  # 1 to 4, intra-cell
    H[3, 2] = t_1 * e_ika1  # 3 to 4, minus-a1 side neighbor
    H[3, 4] = t_1  # 5 to 4, intra-cell

    H[4, 1] = t_1 * e_ika1 * e_mi_ka2  # 2 to 5, a2-a1 side neighbor
    H[4, 3] = t_1  # 4 to 5, intra-cell
    H[4, 5] = t_1  # 6 to 5, intra-cell

    H[5, 0] = t_1 * e_mi_ka2  # 1 to 6, minus-a2 side neighbor
    H[5, 2] = t_1  # 3 to 6, intra-cell
    H[5, 4] = t_1  # 5 to 6, intra-cell
    # print(f"t2, {t_2}")

    # Next near neighbor hopping
    H[0, 2] = (t_2 * e_ika1 + t_2 * e_ika2 + t_2) * e_mi_phi  # 3 to 1, a1, a2, intra-cell
    H[0, 4] = t_2 * e_i_phi * (e_ika2
                               + e_ika2 * e_mi_ka1
                               + 1)  # 5 to 1, a2, a1-a1, intra-cell

    H[1, 3] = t_2 * e_i_phi * (1
                               + e_ika2 * e_mi_ka1
                               + e_mi_ka1)  # 4 to 2, intra-cell, a2-a1, -a1
    H[1, 5] = t_2 * e_mi_phi * (1
                                + e_ika2
                                + e_ika2 * e_mi_ka1)  # 6 to 2, intra-cell, a2, a2-a1

    H[2, 0] = t_2 * e_i_phi * (1 +
                               e_mi_ka2 +
                               e_mi_ka1)  # 1 to 3, intra-cell, a1, a2
    H[2, 4] = t_2 * e_mi_phi * (1 +
                                e_mi_ka1 +
                                e_ika2 * e_mi_ka1)  # 5 to 3, intra-cell, a1-a2, a1

    H[3, 1] = t_2 * e_mi_phi * (1
                                + e_ika1 * e_mi_ka2
                                + e_ika1)  # 2 to 4, -a1, intra-cell, a2-a1
    H[3, 5] = t_2 * e_i_phi * (1
                               + e_ika1
                               + e_ika2)  # 6 to 4, -a1, -a2, intra-cell

    H[4, 0] = t_2 * e_mi_phi * (e_ika1 * e_mi_ka2
                                + 1
                                + e_mi_ka2)  # 1 to 5, a2-a1, intra-cell, a2
    H[4, 2] = t_2 * e_i_phi * (e_ika1
                               + 1
                               + e_ika1 * e_mi_ka2)  # 3 to 5, a2, intra-cell, a1-a2

    H[5, 1] = t_2 * e_i_phi * (e_mi_ka2
                               + 1
                               + e_ika1 * e_mi_ka2)  # 2 to 6, a2, intra-cell, a2
    H[5, 3] = t_2 * e_mi_phi * (e_mi_ka2
                                + 1
                                + e_mi_ka1)  # 4 to 6, -a1, intra-cell, a2-a1
    # print(H)
    print(ishermitian(H, atol=1e-4))
    # if not ishermitian(H,atol=1e-4):
    #     print("H is not hermitian")
    #     break
    # print((H == H.T.conjugate()).all() == True)
    eigval, eigvec = eigh(H)
    all_eigv[:, k_ind] = eigval

E_band = np.array(all_eigv)
plt.figure(1, figsize=(10, 8))
plt.plot(k_path, E_band.T)
plt.xticks(Node, ["G", "K", "M", "K'", "G"])
plt.title(f"Haldane model, t1={t_1}, t2={t_2}, phi={phi:.2f}, tilting={tilting}")
plt.show()
