import numpy as np
from scipy.linalg import ishermitian, eigh, norm
from pymatgen.core import Structure
# from pymatgen.vis import structure_vtk
from Haldane_collection.Lattice import Lattice_2D as lat
# import tqdm
from time import time
import matplotlib.pyplot as plt
import matplotlib
from Sk_tb import Bloch_H, determine_fermi, unit_cell_cutoff
from pathos.multiprocessing import ProcessingPool as Pool

matplotlib.use('TkAgg', force=True)

n_k = 20

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
    # structure_vtk.StructureVis(tbg).show()
    all_eigv = np.zeros((len(lattice_tbg.k_point_path), tbg.num_sites), dtype=np.double)
    all_eigvec = np.zeros((len(lattice_tbg.k_point_path), tbg.num_sites, tbg.num_sites), dtype=complex)
    all_cell_vectors = np.array([a1_index * tbg.lattice.matrix[0][:2] +
                                 a2_index * tbg.lattice.matrix[1][:2]
                                 for a1_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)
                                 for a2_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)])
    for sigma in np.arange(0.4, 30.0, 0.1):
        for k_ind, k in enumerate(lattice_tbg.k_point_path[:]):
            # print("k", k_ind, k)
            H = Bloch_H(tmp_k=k, tmp_tbg=tbg, tmp_all_cell_vectors=all_cell_vectors, tmp_sigma=sigma)
            e, w = eigh(H)
            all_eigv[k_ind, :] = e
            all_eigvec[k_ind, :, :] = w
        t_stop = time()
        print(f"Time: {(t_stop - t_start):.2f}")
        fig = plt.figure()
        ylim = 3

        fermi_level = determine_fermi(N_sample=30, fermi_Nband=14, tmp_tbg=tbg, tmp_lattice_tbg=lattice_tbg,
                                      tmp_sigma=sigma, tmp_all_cell_vectors=all_cell_vectors)
        all_eigv = all_eigv - fermi_level
        plt.plot(lattice_tbg.k_path, all_eigv)
        plt.plot(lattice_tbg.k_path, 0 * np.ones_like(lattice_tbg.k_path), 'k--')

        # plt.plot(lattice_tbg.k_path, fermi_level * np.ones_like(lattice_tbg.k_path), 'k--')
        # plt.ylim(-ylim, ylim)
        plt.xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
        plt.title(f"sigma = {sigma:.4f}")
        plt.savefig(f"./TBG_tb/with_fermi2/TBG_1_{sigma:.4f}.png")
        plt.close(fig)

        fig, axs = plt.subplots(4, 7, figsize=(70, 40))
        for ax_ind, ax in enumerate(axs.ravel()):
            for band_ind, band in enumerate(all_eigv.T):
                for k_ind, k in enumerate(lattice_tbg.k_path):
                    atoms_12_vec = [26, 11, 19, 4, 25, 10, 16, 6, 23, 13, 17, 0]
                    norm_val = norm(all_eigvec[k_ind, atoms_12_vec, band_ind])
                    prob = norm_val ** 2
                    if prob > 0.8:
                        ax.plot(k, band[k_ind], 'bo')
                        print(
                            f"k_ind = {k_ind}, band_ind = {band_ind}, atoms_12_sum = {prob:.2f}, {norm(all_eigvec[k_ind, :, band_ind])}")
                    # print(f"alpha = {np.abs(all_eigvec[k_ind, ax_ind, band_ind])}")
                    ax.plot(k, band[k_ind], 'ro', alpha=np.abs(all_eigvec[k_ind, ax_ind, band_ind]) ** 2)
            ax.set_title(f"atom {ax_ind} projection")
            ax.set_xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
            # ax.set_ylim(-2, 2)
        plt.savefig(f"./TBG_tb/with_fermi2/proj_TBG_1_{sigma:.4f}.png")
        plt.close(fig)
