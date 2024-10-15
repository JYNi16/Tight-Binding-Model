import numpy as np
from Lattice import graphene_60_lat as lat
from scipy.linalg import ishermitian
from Haldane_collection.plot_band import plot_band
import matplotlib

matplotlib.use('TkAgg', force=True)

size_super_cell = 3  # 2x2 supercell
t1 = 1
t2 = 0.15
phi = np.pi / 2

N_atom_unit_cell = 2
N_atom_super_cell = N_atom_unit_cell * size_super_cell ** 2


def get_cell_ind(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    index_a2 = tmp_atom_ind // (N_atom_unit_cell * size_super_cell)
    index_a1 = (tmp_atom_ind % (N_atom_unit_cell * size_super_cell)) // N_atom_unit_cell
    sub_lattice = ["A", "B"][int(tmp_atom_ind % 2)]
    return index_a1, index_a2, sub_lattice


def cell_ind_to_atom_ind(tmp_index_a1, tmp_index_a2, sublat):
    tmp_index_a1, tmp_index_a2 = tmp_index_a1 % size_super_cell, tmp_index_a2 % size_super_cell
    assert tmp_index_a1 in range(size_super_cell)
    assert tmp_index_a2 in range(size_super_cell)
    assert sublat in ['A', 'B']
    if sublat == 'A':
        return (tmp_index_a2 * N_atom_unit_cell * size_super_cell +
                tmp_index_a1 * N_atom_unit_cell)
    if sublat == 'B':
        return (tmp_index_a2 * N_atom_unit_cell * size_super_cell +
                tmp_index_a1 * N_atom_unit_cell + 1)


def find_NN(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    index_a1, index_a2, sub_lat = get_cell_ind(tmp_atom_ind)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        intra_cell_neighbor = tmp_atom_ind + 1
        minus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 - 1),
                                                 tmp_index_a2=index_a2,
                                                 sublat='B')
        minus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                 tmp_index_a2=(index_a2 - 1),
                                                 sublat='B')
        return [intra_cell_neighbor, minus_a1_neighbor, minus_a2_neighbor]
    if tmp_atom_ind % 2 == 1:  # B sublattice
        intra_cell_neighbor = tmp_atom_ind - 1
        plus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 + 1),
                                                tmp_index_a2=index_a2,
                                                sublat='A')
        plus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                tmp_index_a2=(index_a2 + 1),
                                                sublat='A')
        return [intra_cell_neighbor, plus_a1_neighbor, plus_a2_neighbor]


def find_NNN(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    tmp_index_a1, tmp_index_a2, tmp_sub_lattice = get_cell_ind(tmp_atom_ind)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           sublat='A')
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           sublat='A')
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                sublat='A')
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             sublat='A')
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             sublat='A')
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                sublat='A')
        result = {"index": [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor],
                  "phase": np.array([1, -1, 1, -1, 1, -1])}
        return result

    if tmp_atom_ind % 2 == 1:  # B sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           sublat='B')
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           sublat='B')
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                sublat='B')
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             sublat='B')
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             sublat='B')
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                sublat='B')
        result = {"index": [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor],
                  "phase": np.array([-1, 1, -1, 1, -1, 1])}
        return result

if __name__ == "__main__":
    for i in range(N_atom_super_cell):
        print("i", i, "find_NN", find_NN(i), "find_NNN", find_NNN(i))

    all_eigv = np.zeros((N_atom_super_cell, len(lat.k_point_path)), dtype=np.double)

    for k_ind, k in enumerate(lat.k_point_path[:]):
        # print("k", k_ind, k)
        H = np.zeros((N_atom_super_cell, N_atom_super_cell), dtype=complex)
        for atom_ind in range(N_atom_super_cell):
            index_b1, index_b2, sub_lattice = get_cell_ind(atom_ind)
            neighbor_NN = find_NN(atom_ind)
            neighbor_NNN = find_NNN(atom_ind)
            # if size_super_cell == 1:
            #     print("size_super_cell=1")
            #     if sub_lattice == "A":  # neighbors are the same atom
            #         H[atom_ind, neighbor[0]] = t1 * (1 +
            #                                          np.exp(1j * k @ lat.a_1) +
            #                                          np.exp(1j * k @ lat.a_2))
            #     if sub_lattice == "B":
            #         H[atom_ind, neighbor[0]] = t1 * (1 +
            #                                          np.exp(-1j * k @ lat.a_1) +
            #                                          np.exp(-1j * k @ lat.a_2))

            if sub_lattice == "A":  # A sub lattice
                # 3 nearest neighbors
                H[atom_ind, neighbor_NN[0]] = t1
                H[atom_ind, neighbor_NN[1]] = t1 * np.exp(1j * k @ lat.a_1)
                H[atom_ind, neighbor_NN[2]] = t1 * np.exp(1j * k @ lat.a_2)
                # 6 next nearest neighbors
                # print("k", k, "lat.a_1", lat.a_1, "lat.a_2", lat.a_2, "lat.a_2-lat.a_1", lat.a_2 - lat.a_1)
                # print(f"lat.a_1={np.exp(1j * k @ lat.a_1)}, lat.a_2={np.exp(1j * k @ lat.a_2)}, lat.a_2-lat.a_1={np.exp(1j * k @ (lat.a_2 - lat.a_1))}")
                H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.exp(neighbor_NNN["phase"][0] * 1j * phi) * np.exp(
                    1j * k @ -lat.a_1)
                H[atom_ind, neighbor_NNN["index"][1]] = t2 * np.exp(neighbor_NNN["phase"][1] * 1j * phi) * np.exp(
                    1j * k @ -lat.a_2)
                H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.exp(neighbor_NNN["phase"][2] * 1j * phi) * np.exp(
                    1j * k @ (lat.a_1 - lat.a_2))
                H[atom_ind, neighbor_NNN["index"][3]] = t2 * np.exp(neighbor_NNN["phase"][3] * 1j * phi) * np.exp(
                    1j * k @ lat.a_1)
                H[atom_ind, neighbor_NNN["index"][4]] = t2 * np.exp(neighbor_NNN["phase"][4] * 1j * phi) * np.exp(
                    1j * k @ lat.a_2)
                H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.exp(neighbor_NNN["phase"][5] * 1j * phi) * np.exp(
                    1j * k @ (lat.a_2 - lat.a_1))

            if sub_lattice == "B":  # B sub lattice
                # 3 nearest neighbors
                H[atom_ind, neighbor_NN[0]] = t1
                H[atom_ind, neighbor_NN[1]] = t1 * np.exp(-1j * k @ lat.a_1)
                H[atom_ind, neighbor_NN[2]] = t1 * np.exp(-1j * k @ lat.a_2)
                # 6 next nearest neighbors
                H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.exp(neighbor_NNN["phase"][0] * 1j * phi) * np.exp(
                    1j * k @ -lat.a_1)
                H[atom_ind, neighbor_NNN["index"][1]] = t2 * np.exp(neighbor_NNN["phase"][1] * 1j * phi) * np.exp(
                    1j * k @ -lat.a_2)
                H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.exp(neighbor_NNN["phase"][2] * 1j * phi) * np.exp(
                    1j * k @ (lat.a_1 - lat.a_2))
                H[atom_ind, neighbor_NNN["index"][3]] = t2 * np.exp(neighbor_NNN["phase"][3] * 1j * phi) * np.exp(
                    1j * k @ lat.a_1)
                H[atom_ind, neighbor_NNN["index"][4]] = t2 * np.exp(neighbor_NNN["phase"][4] * 1j * phi) * np.exp(
                    1j * k @ lat.a_2)
                H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.exp(neighbor_NNN["phase"][5] * 1j * phi) * np.exp(
                    1j * k @ (lat.a_2 - lat.a_1))

        assert ishermitian(H, atol=1e-8)
        # if k_ind==3:
        #     break
        all_eigv[:, k_ind] = np.linalg.eigvalsh(H)

    # Plot the band structure
    plot_band(all_eigv, lat)
    # plt.show()
