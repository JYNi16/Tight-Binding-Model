import numpy as np
from config import *
import matplotlib.pyplot as plt

# W = 50  #width of the ribbon
#
# t_1 = 1
# t_2 = 0.1
# phi = np.pi / 4

M = 0.2
W = 50  #width of the ribbon
t_1 = 1
t_2 = 0.1842
phi = np.pi / 2

tilting = -0.5

a = np.array([0, 1])

# k_x = np.array([0, 1])
Nk = 50
k_x_lin = np.linspace(-np.pi, np.pi, Nk)
# all_kx = np.array([np.zeros(50), k_x_lin])

all_eigv = np.zeros((W * 2, Nk), dtype=np.double)
for k_ind, k in enumerate(k_x_lin):
    k_x = np.array([0, k])
    Ribbon_ham = np.zeros((2 * W, 2 * W), dtype=complex)

    e_i_ka = np.exp(1j * k_x @ a)
    e_mi_ka = np.exp(-1j * k_x @ a)

    e_i_phi = np.exp(1j * phi)
    e_mi_phi = np.exp(-1j * phi)

    for i in range(W):
        for j in range(W):
            if i == j:  # Inside unit cell
                Ribbon_ham[
                    2 * i, 2 * j] = t_2 * e_mi_ka * e_mi_ka + t_2 * e_i_phi * e_i_ka + M + tilting * k  # 1 to 1, two neighbors
                Ribbon_ham[2 * i, 2 * j + 1] = t_1 * e_i_ka + t_1  # 2 to 1, intracell and down-side neighbor
                Ribbon_ham[2 * i + 1, 2 * j] = t_1 * e_mi_ka + t_1  # 1 to 2, intracell and up-side neighbor
                Ribbon_ham[
                    2 * i + 1, 2 * j + 1] = t_2 * e_i_phi * e_mi_ka + t_2 * e_mi_phi * e_i_ka - M + tilting * k  # 2 to 2, two neighbors
            elif i == j + 1:
                Ribbon_ham[
                    2 * i, 2 * j] = t_2 * e_mi_phi + t_2 * e_i_phi * e_mi_ka  # 1 to 3, intracell and up-side neighbor
                Ribbon_ham[2 * i, 2 * j + 1] = t_1  # 2 to 3, only intra-cell
                Ribbon_ham[2 * i + 1, 2 * j] = 0  # 1 to 4, no hopping
                Ribbon_ham[
                    2 * i + 1, 2 * j + 1] = t_2 * e_i_phi + t_2 * e_mi_phi * e_mi_ka  # 2 to 4, intracell and up-side neighbor

                # hermitian
                Ribbon_ham[2 * j, 2 * i] = Ribbon_ham[2 * i, 2 * j].conj()
                Ribbon_ham[2 * j + 1, 2 * i] = Ribbon_ham[2 * i, 2 * j + 1].conj()
                Ribbon_ham[2 * j, 2 * i + 1] = Ribbon_ham[2 * i + 1, 2 * j].conj()
                Ribbon_ham[2 * j + 1, 2 * i + 1] = Ribbon_ham[2 * i + 1, 2 * j + 1].conj()

    eigval, eigvec = np.linalg.eigh(Ribbon_ham)
    all_eigv[:, k_ind] = eigval

plt.plot(k_x_lin, all_eigv.T)
title = f"ZigZag with M={M}, W={W}, t1={t_1}, t2={t_2:.2f}, phi={phi:.2f}, tilting={tilting}"
plt.title(title)
plt.xlabel("k_y")
plt.show()
