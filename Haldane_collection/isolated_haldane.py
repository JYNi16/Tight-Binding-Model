import numpy as np
from Lattice import graphene_60_lat as lat
from scipy.linalg import ishermitian
# from Haldane_collection.plot_band import plot_band
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg', force=True)

size_super_cell = 20  # 2x2 supercell
t1 = 1
t2 = 0.05
phi = np.pi / 2

N_atom_unit_cell = 2
N_atom_super_cell = N_atom_unit_cell * size_super_cell ** 2
eta = 0.5
xrange = 50
y_range = 0.5


def get_cell_ind(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    index_a2 = tmp_atom_ind // (N_atom_unit_cell * size_super_cell)
    index_a1 = (tmp_atom_ind % (N_atom_unit_cell * size_super_cell)) // N_atom_unit_cell
    sub_lattice = ["A", "B"][int(tmp_atom_ind % 2)]
    return index_a1, index_a2, sub_lattice


def cell_ind_to_atom_ind(tmp_index_a1, tmp_index_a2, tmp_sublat):
    # tmp_index_a1, tmp_index_a2 = tmp_index_a1 % size_super_cell, tmp_index_a2 % size_super_cell
    if tmp_index_a1 < 0 or tmp_index_a2 < 0 or tmp_index_a1 >= size_super_cell or tmp_index_a2 >= size_super_cell:
        return -1

    assert tmp_index_a1 in range(size_super_cell)
    assert tmp_index_a2 in range(size_super_cell)
    assert tmp_sublat in ['A', 'B']
    if tmp_sublat == 'A':
        return (tmp_index_a2 * N_atom_unit_cell * size_super_cell +
                tmp_index_a1 * N_atom_unit_cell)
    if tmp_sublat == 'B':
        return (tmp_index_a2 * N_atom_unit_cell * size_super_cell +
                tmp_index_a1 * N_atom_unit_cell + 1)


def find_NN(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    index_a1, index_a2, sub_lat = get_cell_ind(tmp_atom_ind)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        intra_cell_neighbor = tmp_atom_ind + 1  # s
        minus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 - 1),
                                                 tmp_index_a2=index_a2,
                                                 tmp_sublat='B')  # w
        minus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                 tmp_index_a2=(index_a2 - 1),
                                                 tmp_sublat='B')  # w
        return [intra_cell_neighbor, minus_a1_neighbor, minus_a2_neighbor]
    if tmp_atom_ind % 2 == 1:  # B sublattice
        intra_cell_neighbor = tmp_atom_ind - 1  # s
        plus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 + 1),
                                                tmp_index_a2=index_a2,
                                                tmp_sublat='A')  # w
        plus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                tmp_index_a2=(index_a2 + 1),
                                                tmp_sublat='A')  # w
        return [intra_cell_neighbor, plus_a1_neighbor, plus_a2_neighbor]


def find_NNN(tmp_atom_ind):
    assert tmp_atom_ind in range(N_atom_super_cell)
    tmp_index_a1, tmp_index_a2, tmp_sub_lattice = get_cell_ind(tmp_atom_ind)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           tmp_sublat='A')  # s
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           tmp_sublat='A')  # s
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                tmp_sublat='A')  # w
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             tmp_sublat='A')  # s
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             tmp_sublat='A')  # w
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                tmp_sublat='A')  # w
        result = {"index": [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor],
                  "phase": np.array([1, -1, 1, -1, 1, -1])}
        return result

    if tmp_atom_ind % 2 == 1:  # B sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           tmp_sublat='B')
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           tmp_sublat='B')
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                tmp_sublat='B')
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             tmp_sublat='B')
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             tmp_sublat='B')
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                tmp_sublat='B')
        result = {"index": [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor],
                  "phase": np.array([-1, 1, -1, 1, -1, 1])}
        return result


for i in range(N_atom_super_cell):
    print("i", i, "find_NN", find_NN(i), "find_NNN", find_NNN(i))

all_eigv = np.zeros((N_atom_super_cell, len(lat.k_point_path)), dtype=np.double)

k = np.array([0, 0])
H = np.zeros((N_atom_super_cell, N_atom_super_cell), dtype=complex)

for atom_ind in range(N_atom_super_cell):
    index_b1, index_b2, sub_lattice = get_cell_ind(atom_ind)
    neighbor_NN = find_NN(atom_ind)
    neighbor_NNN = find_NNN(atom_ind)

    if sub_lattice == "A":  # A sub lattice
        # 3 nearest neighbors
        if neighbor_NN[0] != -1:
            H[atom_ind, neighbor_NN[0]] = t1
        if neighbor_NN[1] != -1:
            H[atom_ind, neighbor_NN[1]] = t1 * np.exp(1j * k @ lat.a_1) * eta
        if neighbor_NN[2] != -1:
            H[atom_ind, neighbor_NN[2]] = t1 * np.exp(1j * k @ lat.a_2) * eta
        # 6 next nearest neighbors
        # print("k", k, "lat.a_1", lat.a_1, "lat.a_2", lat.a_2, "lat.a_2-lat.a_1", lat.a_2 - lat.a_1)
        # print(f"lat.a_1={np.exp(1j * k @ lat.a_1)}, lat.a_2={np.exp(1j * k @ lat.a_2)}, lat.a_2-lat.a_1={np.exp(1j * k @ (lat.a_2 - lat.a_1))}")
        if neighbor_NNN["index"][0] != -1:
            # H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.exp(neighbor_NNN["phase"][0] * 1j * phi) * np.exp(
            #     1j * k @ -lat.a_1)
            H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.sin(neighbor_NNN["phase"][0] * phi) * np.sin(k @ lat.a_1)

        if neighbor_NNN["index"][1] != -1:
            # H[atom_ind, neighbor_NNN["index"][1]] = t2 * np.exp(neighbor_NNN["phase"][1] * 1j * phi) * np.exp(
            #     1j * k @ -lat.a_2)
            H[atom_ind, neighbor_NNN["index"][1]] = t2 * np.sin(neighbor_NNN["phase"][1] * phi) * np.sin(k @ -lat.a_2)

        if neighbor_NNN["index"][2] != -1:
            # H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.exp(neighbor_NNN["phase"][2] * 1j * phi) * np.exp(
            #     1j * k @ (lat.a_1 - lat.a_2)) * eta
            H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.sin(neighbor_NNN["phase"][2] * phi) * np.sin(
                k @ (lat.a_1 - lat.a_2)) * eta
            # t2 * np.exp(neighbor_NNN["phase"][2] * 1j * phi) * np.exp(
            # 1j * k @ (lat.a_1 - lat.a_2)) * eta)
        if neighbor_NNN["index"][3] != -1:
            H[atom_ind, neighbor_NNN["index"][3]] = t2 * np.sin(neighbor_NNN["phase"][3] * phi) * np.sin(k @ lat.a_1)
            # t2 * np.exp(neighbor_NNN["phase"][3] * 1j * phi) * np.exp(
            # 1j * k @ lat.a_1))
        if neighbor_NNN["index"][4] != -1:
            H[atom_ind, neighbor_NNN["index"][4]] = t2 * np.sin(neighbor_NNN["phase"][4] * phi) * np.sin(k @ -lat.a_2)

            # H[atom_ind, neighbor_NNN["index"][4]] = t2 * np.exp(neighbor_NNN["phase"][4] * 1j * phi) * np.exp(
            #     1j * k @ lat.a_2)
        if neighbor_NNN["index"][5] != -1:
            H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.sin(neighbor_NNN["phase"][5] * phi) * np.sin(
                k @ (lat.a_2 - lat.a_1)) * eta
            # H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.exp(neighbor_NNN["phase"][5] * 1j * phi) * np.exp(
            #     1j * k @ (lat.a_2 - lat.a_1)) * eta

    if sub_lattice == "B":  # B sub lattice
        # 3 nearest neighbors
        if neighbor_NN[0] != -1:
            H[atom_ind, neighbor_NN[0]] = t1
        if neighbor_NN[1] != -1:
            H[atom_ind, neighbor_NN[1]] = t1 * np.exp(-1j * k @ lat.a_1) * eta
        if neighbor_NN[2] != -1:
            H[atom_ind, neighbor_NN[2]] = t1 * np.exp(-1j * k @ lat.a_2) * eta
        # 6 next nearest neighbors
        if neighbor_NNN["index"][0] != -1:
            H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.sin(neighbor_NNN["phase"][0] * phi) * np.sin(k @ -lat.a_1)
            # H[atom_ind, neighbor_NNN["index"][0]] = t2 * np.exp(neighbor_NNN["phase"][0] * 1j * phi) * np.exp(
            #     1j * k @ -lat.a_1)
        if neighbor_NNN["index"][1] != -1:
            H[atom_ind, neighbor_NNN["index"][1]] = -t2 * np.sin(neighbor_NNN["phase"][1] * phi) * np.sin(k @ -lat.a_2)
            # H[atom_ind, neighbor_NNN["index"][1]] = t2 * np.exp(neighbor_NNN["phase"][1] * 1j * phi) * np.exp(
            #     1j * k @ -lat.a_2)
        if neighbor_NNN["index"][2] != -1:
            H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.sin(neighbor_NNN["phase"][2] * phi) * np.sin(
                k @ (lat.a_1 - lat.a_2)) * eta
            # H[atom_ind, neighbor_NNN["index"][2]] = t2 * np.exp(neighbor_NNN["phase"][2] * 1j * phi) * np.exp(
            #     1j * k @ (lat.a_1 - lat.a_2)) * eta
        if neighbor_NNN["index"][3] != -1:
            H[atom_ind, neighbor_NNN["index"][3]] = t2 * np.sin(neighbor_NNN["phase"][3] * phi) * np.sin(k @ lat.a_1)
            # H[atom_ind, neighbor_NNN["index"][3]] = t2 * np.exp(neighbor_NNN["phase"][3] * 1j * phi) * np.exp(
            #     1j * k @ lat.a_1)
        if neighbor_NNN["index"][4] != -1:
            H[atom_ind, neighbor_NNN["index"][4]] = -t2 * np.sin(neighbor_NNN["phase"][4] * phi) * np.sin(k @ lat.a_2)
            # H[atom_ind, neighbor_NNN["index"][4]] = t2 * np.exp(neighbor_NNN["phase"][4] * 1j * phi) * np.exp(
            #     1j * k @ lat.a_2)
        if neighbor_NNN["index"][5] != -1:
            H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.sin(neighbor_NNN["phase"][5] * phi) * np.sin(
                k @ (lat.a_2 - lat.a_1)) * eta
            # H[atom_ind, neighbor_NNN["index"][5]] = t2 * np.exp(neighbor_NNN["phase"][5] * 1j * phi) * np.exp(
            #     1j * k @ (lat.a_2 - lat.a_1)) * eta

assert ishermitian(H, atol=1e-8)

eigv = np.linalg.eigvalsh(H)
plt.plot(eigv, "bo")
plt.ylim(-y_range, y_range)
# plt.xlim(N_atom_super_cell/2-xrange,N_atom_super_cell/2+xrange)
plt.plot([0, N_atom_super_cell], [0, 0], "r--")

# Plot the band structure
# plot_band(all_eigv, lat)
plt.show()
