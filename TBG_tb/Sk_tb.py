import numpy as np
from scipy.linalg import ishermitian, eigh, norm
from pymatgen.core import Structure

from Haldane_collection.Lattice import Lattice_2D as lat
import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg', force=True)

unit_cell_cutoff = 1

d = 3.349  # between two atoms in the same layer
# d2 = d ** 2
ac = 1.418
a0 = ac * np.sqrt(3)

r_0 = 0.184 * a0  # decaying length

sigma = 14.9

n_k = 100


def interlayer_hopping(r_xy, tmp_d, v_sigma):
    xy_length = np.linalg.norm(r_xy)
    tmp_d2 = tmp_d ** 2
    xy2 = xy_length ** 2
    r2 = xy2 + tmp_d2
    r = np.sqrt(r2)

    cos2 = tmp_d2 / r2
    sin2 = 1 - cos2

    hopping = (sin2 * (-2.7) * np.exp(-((r - ac) / r_0)) +
               cos2 * v_sigma * np.exp(-((r - d) / r_0)))

    return hopping


def h_ij(tmp_i, tmp_j, tmp_phases, tmp_sigma, tmp_tbg, tmp_all_cell_vectors):
    tmp_i, tmp_j = tmp_tbg.sites[tmp_i], tmp_tbg.sites[tmp_j]
    all_paras = zip(tmp_all_cell_vectors + (tmp_i.coords[:2] - tmp_j.coords[:2]),
                    list([tmp_i.coords[2] - tmp_j.coords[2]]) * len(tmp_all_cell_vectors),
                    list([tmp_sigma]) * len(tmp_all_cell_vectors))
    all_hopping = list(map(lambda x: interlayer_hopping(*x), all_paras))
    tmp2 = all_hopping @ tmp_phases
    return tmp2


def Bloch_H(tmp_k, tmp_tbg, tmp_all_cell_vectors, tmp_sigma=0.48):
    H = np.zeros((tmp_tbg.num_sites, tmp_tbg.num_sites), dtype=complex)
    for i, site_i in enumerate(tmp_tbg.sites):
        for j, site_j in enumerate(tmp_tbg.sites):
            if i == j:
                H[i, j] = -0.78
                continue
            phases = np.exp(1j * tmp_all_cell_vectors @ tmp_k)
            H[i, j] = h_ij(i, j, phases, tmp_sigma, tmp_tbg, tmp_all_cell_vectors)
    assert ishermitian(H, atol=1e-8) is True
    return H


def determine_fermi(tmp_tbg, tmp_lattice_tbg, tmp_all_cell_vectors, tmp_sigma, N_sample=20, fermi_Nband=14):
    x = np.linspace(-0.5, 0.5, N_sample)
    y = np.linspace(-0.5, 0.5, N_sample)
    mesh = np.meshgrid(x, y)
    N_states = N_sample ** 2 * tmp_tbg.num_sites
    filled_N = N_sample ** 2 * fermi_Nband
    k_mesh = tmp_lattice_tbg.rlat[:2, :2] @ np.array(mesh).reshape(2, -1)
    all_eigen = np.array(list(map(lambda k: np.linalg.eigvalsh(Bloch_H(k, tmp_tbg=tmp_tbg, tmp_all_cell_vectors=tmp_all_cell_vectors, tmp_sigma=tmp_sigma)), k_mesh.T))).reshape(N_states, -1)
    all_eigen = np.sort(all_eigen, axis=0)
    fermi_level = all_eigen[filled_N, 0]
    return fermi_level


if __name__ == '__main__':
    t_start = time()
    # tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_1.vasp')
    tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_1_hex.vasp')
    # tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_3.vasp')
    # tbg = Structure.from_file("./TBG_tb/POSCAR/graphene_60_hex_orgin.vasp")
    lattice_tbg = lat(lattice_vector=tbg.lattice.matrix.T,
                      theta=60,
                      # high_symmetry_points=np.array([[0, 0],  # Gamma
                      #                                [1 / 3, 2 / 3],  # K
                      #                                [1 / 2, 1 / 2],  # M
                      #                                [2 / 3, 1 / 3],  # K_prime
                      #                                [0, 0]], dtype=np.float64).T,
                      high_symmetry_points=np.array([[1 / 3, 2 / 3],  # K
                                                     [0, 0],
                                                     [1 / 2, 1 / 2],  # M
                                                     [2 / 3, 1 / 3]], dtype=np.float64).T,
                      high_symmetry_points_labels=['K', '$\Gamma$', 'M', '$K\'$'],
                      # high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'],
                      Nk=n_k)
    all_eigv = np.zeros((len(lattice_tbg.k_point_path), tbg.num_sites), dtype=np.double)

    all_eigvec = np.zeros((len(lattice_tbg.k_point_path), tbg.num_sites, tbg.num_sites), dtype=complex)

    all_cell_vectors = np.array([a1_index * tbg.lattice.matrix[0][:2] +
                                 a2_index * tbg.lattice.matrix[1][:2]
                                 for a1_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)
                                 for a2_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)])

    for k_ind, k in enumerate(tqdm.tqdm(lattice_tbg.k_point_path[:])):
        H = Bloch_H(tmp_k=k, tmp_tbg=tbg, tmp_all_cell_vectors=all_cell_vectors, tmp_sigma=sigma)
        e, w = eigh(H)
        all_eigv[k_ind, :] = e
        all_eigvec[k_ind, :, :] = w
    t_stop = time()
    print(f"Time: {(t_stop - t_start):.2f}")
    fermi_level = determine_fermi(N_sample=30, fermi_Nband=14, tmp_tbg=tbg, tmp_lattice_tbg=lattice_tbg,
                                  tmp_all_cell_vectors=all_cell_vectors, tmp_sigma=sigma)
    t_stop = time()
    print(f"Time: {(t_stop - t_start):.2f}")
    # fig = plt.figure(figsize=(40, 70))

    # fig, axs = plt.subplots(4, 7,figsize=(70, 40))
    #
    # for ax_ind, ax in enumerate(axs.ravel()):
    #     for band_ind, band in enumerate(all_eigv.T):
    #         for k_ind, k in enumerate(lattice_tbg.k_path):
    #             print(f"alpha = {np.abs(all_eigvec[k_ind, ax_ind, band_ind])}")
    #             ax.plot(k, band[k_ind], 'ro', alpha=np.abs(all_eigvec[k_ind, ax_ind, band_ind]) ** 2)
    #     ax.set_title(f"atom {ax_ind} projection")
    #     ax.set_xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
    #     ax.set_ylim(-2, 2)
    # plt.savefig(f"TBG_1_{sigma:.4f}.png")

    # ax.set_title("s orbital")



    # # fig = plt.figure()
    ylim = 5.0
    all_eigv = all_eigv - fermi_level
    plt.plot(lattice_tbg.k_path, all_eigv)
    plt.plot(lattice_tbg.k_path, 0 * np.ones_like(lattice_tbg.k_path), 'k--')
    for band_ind, band in enumerate(all_eigv.T):
        for k_ind, k in enumerate(lattice_tbg.k_path):
            atoms_12_vec = [26, 11, 19, 4, 25, 10, 16, 6, 23, 13, 17, 0]
            norm_val = norm(all_eigvec[k_ind, atoms_12_vec, band_ind])
            prob = norm_val ** 2
            if prob > 0.8:
                plt.plot(k, band[k_ind], 'bo')
                print(
                    f"k_ind = {k_ind}, band_ind = {band_ind}, atoms_12_sum = {prob:.2f}, {norm(all_eigvec[k_ind, :, band_ind])}")
    # for ind, band in enumerate(all_eigv.T):
    #     if ind == 12:
    #         plt.plot(lattice_tbg.k_path, band, 'yellow', linewidth=2, label="flat")
    #         continue
    #     if ind == 13:
    #         plt.plot(lattice_tbg.k_path, band, 'r', linewidth=2, label="top of valence band")
    #         continue
    #     if ind == 14:
    #         plt.plot(lattice_tbg.k_path, band, 'b', linewidth=2, label="bottom of conduction band")
    #         continue
    #
    #     plt.plot(lattice_tbg.k_path, band, 'k')
    #     print()
    # flat_bands = all_eigvec[:, :, 12]
    # atoms = np.argsort(np.abs(flat_bands), -1)
    #
    plt.ylim(-ylim, ylim)
    plt.xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
    plt.title(f"sigma = {sigma:.4f}")
    plt.savefig(f"./TBG_tb/pic/TBG_1_{sigma:.4f}.png")
    # plt.close(fig)
    plt.show()
