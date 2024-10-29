import numpy as np
from scipy.linalg import ishermitian, eigh, norm
from pymatgen.core import Structure
from Haldane_collection.Lattice import Lattice_2D as lat
# import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg', force=True)

unit_cell_cutoff = 1

d = 3.349  # between two atoms in the same layer
d2 = d ** 2
ac = 1.418
a0 = ac * np.sqrt(3)

r_0 = 0.184 * a0  # decaying length

q_pi = 3.14
q_sigma = 7.43

n_k = 20


def interlayer_hopping(r_xy, tmp_d):
    xy_length = np.linalg.norm(r_xy)
    r2 = xy_length ** 2 + d2
    r = np.sqrt(r2)

    cos2 = d2 / r2
    sin2 = 1 - cos2

    hopping = (sin2 * (-2.7) * np.exp(-((r - ac) / r_0)) +
               cos2 * 0.48 * np.exp(-((r - np.abs(tmp_d)) / r_0)))

    # hopping = (sin2 * (-2.7) * np.exp(q_pi * (1 - r / ac)) +
    #            cos2 * 0.48 * np.exp(q_sigma * (1 - r / np.abs(tmp_d))))

    return hopping


def h_ij(tmp_i, tmp_j, tmp_phases):
    tmp_i, tmp_j = tbg.sites[tmp_i], tbg.sites[tmp_j]
    all_paras = zip(all_cell_vectors + (tmp_i.coords[:2] - tmp_j.coords[:2]),
                    list([tmp_i.coords[2] - tmp_j.coords[2]]) * len(all_cell_vectors))
    all_hopping = list(map(lambda x: interlayer_hopping(*x), all_paras))
    tmp2 = all_hopping @ tmp_phases
    return tmp2


if __name__ == '__main__':
    t_start = time()
    # tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_1.vasp')
    tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_3.vasp')
    lattice_tbg = lat(lattice_vector=tbg.lattice.matrix.T,
                      theta=60,
                      high_symmetry_points=np.array([[0, 0],  # Gamma
                                                     [1 / 3, 2 / 3],  # K
                                                     [1 / 2, 1 / 2],  # M
                                                     [2 / 3, 1 / 3],  # K_prime
                                                     [0, 0]], dtype=np.float64).T,
                      high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'],
                      Nk=n_k)
    all_eigv = np.zeros((len(lattice_tbg.k_point_path), tbg.num_sites), dtype=np.double)

    all_cell_vectors = np.array([a1_index * tbg.lattice.matrix[0][:2] +
                                 a2_index * tbg.lattice.matrix[1][:2]
                                 for a1_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)
                                 for a2_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)])

    for k_ind, k in enumerate(lattice_tbg.k_point_path[:]):
        print("k", k_ind, k)
        H = np.zeros((tbg.num_sites, tbg.num_sites), dtype=complex)

        for i, site_i in enumerate(tbg.sites):
            for j, site_j in enumerate(tbg.sites):
                if i == j:
                    H[i, j] = -0.78
                phases = np.exp(1j * all_cell_vectors @ k)

                # tmp =
                # for ind, cell_vector in enumerate(all_cell_vectors):
                #     print(f"sites: {site_i.coords, site_j.coords}")
                #     r_ij = (site_i.coords[:2] - site_j.coords[:2] +
                #             cell_vector)
                #     delta_d = site_i.coords[2] - site_j.coords[2]
                #     print(f"r_ij: {r_ij}, delta_d: {delta_d}")
                #     # print(f"a_1, a_2: {a1_index, a2_index}, cell_vector: {cell_vector}, r_ij: {r_ij}")
                #     tmp += interlayer_hopping(r_ij, d=delta_d) * phases[ind]
                # print(f"tmp: {tmp}, tmp2: {tmp2}")
                # assert np.isclose(tmp, tmp2)

                H[i, j] = h_ij(i, j, phases)
        assert ishermitian(H, atol=1e-8) is True
        e, w = eigh(H)
        all_eigv[k_ind, :] = e
    t_stop = time()
    print(f"Time: {(t_stop - t_start):.2f}")
    plt.figure()

    ylim = 0.1

    plt.plot(lattice_tbg.k_path, all_eigv)
    plt.ylim(-ylim, ylim)
    # plt.plot(all_eigv)
    plt.xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
    plt.show()
