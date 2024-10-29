# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:24:55 2024

@author: Curry
"""

import numpy as np


# k_npoints = 20

def Dist(r1, r2):
    return np.linalg.norm(r1 - r2)


def k_point_gen(k_syms, k_npoints=100):
    """
    Parameters
    ----------
    k_syms : list
        high syms point in 1st BZ.
        such as k_syms = [G, K, M,...]
    Returns
    -------
    k_point_path = [kgk, kkm, kmk2, kk2g]
    which kgk represent the kpoints between G and K ..
    """
    k_point_path = []
    for i in range(len(k_syms) - 1):
        k_point_path.append(np.linspace(k_syms[i], k_syms[i + 1], k_npoints))

    return k_point_path


def k_dist_gen(k_syms, k_npoints=100):
    k_dist = []
    for i in range(len(k_syms) - 1):
        k_dist.append(Dist(k_syms[i], k_syms[i + 1]))

    lk = np.linspace(0, 1, k_npoints)
    k_tmp = []
    for j in range(len(k_dist)):
        if j == 0:
            x_tmp = k_dist[j] * lk
            k_tmp.append(x_tmp)
        else:
            x_tmp = k_dist[j] * lk + k_tmp[-1][-1]
            k_tmp.append(x_tmp)

    return k_tmp


def k_path_sym_gen(k_syms, k_npoints=100):
    #k_syms = [cf.G, cf.K, cf.M, cf.K2, cf.G]
    k_point_path = k_point_gen(k_syms, k_npoints=k_npoints)

    k_tmp = k_dist_gen(k_syms, k_npoints=k_npoints)
    k_path = np.array(k_tmp).flatten()

    Node = [0]
    for i in range(len(k_tmp)):
        Node.append(k_tmp[i][-1])

    return k_point_path, k_path, Node
