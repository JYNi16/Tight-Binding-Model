import numpy as np
from scipy.linalg import ishermitian, eigh, norm
from pymatgen.core import Structure
# from pymatgen.vis import structure_vtk
from Haldane_collection.Lattice import Lattice_2D as lat
# import tqdm
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

q_pi = 3.14
q_sigma = 7.43

n_k = 20


def interlayer_hopping(r_xy, tmp_d, v_sigma=0.48):
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


def h_ij(tmp_i, tmp_j, tmp_phases, tmp_sigma):
    tmp_i, tmp_j = tbg.sites[tmp_i], tbg.sites[tmp_j]
    all_paras = zip(all_cell_vectors + (tmp_i.coords[:2] - tmp_j.coords[:2]),
                    list([tmp_i.coords[2] - tmp_j.coords[2]]) * len(all_cell_vectors),
                    list([tmp_sigma]) * len(all_cell_vectors))
    all_hopping = list(map(lambda x: interlayer_hopping(*x), all_paras))
    tmp2 = all_hopping @ tmp_phases
    return tmp2


if __name__ == '__main__':
    t_start = time()
    tbg = Structure.from_file('./TBG_tb/POSCAR/TBG_1.vasp')
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

    all_cell_vectors = np.array([a1_index * tbg.lattice.matrix[0][:2] +
                                 a2_index * tbg.lattice.matrix[1][:2]
                                 for a1_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)
                                 for a2_index in range(-unit_cell_cutoff, unit_cell_cutoff + 1)])
    for sigma in np.arange(0.1, 500, 0.1):
        for k_ind, k in enumerate(lattice_tbg.k_point_path[:]):
            # print("k", k_ind, k)
            H = np.zeros((tbg.num_sites, tbg.num_sites), dtype=complex)

            for i, site_i in enumerate(tbg.sites):
                for j, site_j in enumerate(tbg.sites):
                    if i == j:
                        H[i, j] = -0.78
                        continue
                    phases = np.exp(1j * all_cell_vectors @ k)
                    H[i, j] = h_ij(i, j, phases, sigma)
            assert ishermitian(H, atol=1e-8) is True
            e, w = eigh(H)
            all_eigv[k_ind, :] = e
        t_stop = time()
        print(f"Time: {(t_stop - t_start):.2f}")
        fig = plt.figure()
        ylim = 2
        plt.plot(lattice_tbg.k_path, all_eigv)
        plt.ylim(-ylim, ylim)
        # plt.plot(all_eigv)
        plt.xticks(lattice_tbg.Node, lattice_tbg.high_symmetry_points_labels)
        plt.title(f"sigma = {sigma:.4f}")
        plt.savefig(f"./TBG_tb/pic/TBG_1_{sigma:.4f}.png")
        plt.close(fig)
        # plt.show()
