def get_cell_ind(tmp_atom_ind,  size_super_cell, N_atom_unit_cell=2):

    assert tmp_atom_ind in range(N_atom_unit_cell * size_super_cell * size_super_cell)
    index_a2 = tmp_atom_ind // (N_atom_unit_cell * size_super_cell)
    index_a1 = (tmp_atom_ind % (N_atom_unit_cell * size_super_cell)) // N_atom_unit_cell
    sub_lattice = ["A", "B"][int(tmp_atom_ind % 2)]
    return index_a1, index_a2, sub_lattice


def cell_ind_to_atom_ind(tmp_index_a1, tmp_index_a2, sublat, size_super_cell, N_atom_unit_cell=2):
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


def find_NN(tmp_atom_ind,N_atom_super_cell,size_super_cell=2):
    assert tmp_atom_ind in range(N_atom_super_cell)
    index_a1, index_a2, sub_lat = get_cell_ind(tmp_atom_ind,size_super_cell=size_super_cell)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        intra_cell_neighbor = tmp_atom_ind + 1
        minus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 - 1),
                                                 tmp_index_a2=index_a2,
                                                 sublat='B',
                                                 size_super_cell=size_super_cell)
        minus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                 tmp_index_a2=(index_a2 - 1),
                                                 sublat='B',
                                                 size_super_cell=size_super_cell)
        return [intra_cell_neighbor, minus_a1_neighbor, minus_a2_neighbor]
    if tmp_atom_ind % 2 == 1:  # B sublattice
        intra_cell_neighbor = tmp_atom_ind - 1
        plus_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(index_a1 + 1),
                                                tmp_index_a2=index_a2,
                                                sublat='A',
                                                size_super_cell=size_super_cell)
        plus_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=index_a1,
                                                tmp_index_a2=(index_a2 + 1),
                                                sublat='A',
                                                size_super_cell=size_super_cell)
        return [intra_cell_neighbor, plus_a1_neighbor, plus_a2_neighbor]


def find_NNN(tmp_atom_ind, N_atom_unit_cell=2, size_super_cell=2):
    assert tmp_atom_ind in range(N_atom_unit_cell * size_super_cell * size_super_cell)
    tmp_index_a1, tmp_index_a2, tmp_sub_lattice = get_cell_ind(tmp_atom_ind)
    if tmp_atom_ind % 2 == 0:  # A sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           sublat='A',
                                           size_super_cell=size_super_cell)
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           sublat='A',
                                           size_super_cell=size_super_cell)
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                sublat='A',
                                                size_super_cell=size_super_cell)
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             sublat='A',
                                             size_super_cell=size_super_cell)
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             sublat='A',
                                             size_super_cell=size_super_cell)
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                sublat='A',
                                                size_super_cell=size_super_cell)
        result = [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor]
        return result

    if tmp_atom_ind % 2 == 1:  # B sublattice
        a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                           tmp_index_a2=tmp_index_a2,
                                           sublat='B',
                                           size_super_cell=size_super_cell)
        a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                           tmp_index_a2=(tmp_index_a2 + 1),
                                           sublat='B',
                                           size_super_cell=size_super_cell)
        a2_m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                                tmp_index_a2=(tmp_index_a2 + 1),
                                                sublat='B',
                                                size_super_cell=size_super_cell)
        m_a1_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 - 1),
                                             tmp_index_a2=tmp_index_a2,
                                             sublat='B',
                                             size_super_cell=size_super_cell)
        m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=tmp_index_a1,
                                             tmp_index_a2=(tmp_index_a2 - 1),
                                             sublat='B',
                                             size_super_cell=size_super_cell)
        a1_m_a2_neighbor = cell_ind_to_atom_ind(tmp_index_a1=(tmp_index_a1 + 1),
                                                tmp_index_a2=(tmp_index_a2 - 1),
                                                sublat='B',
                                                size_super_cell=size_super_cell)
        result = [a1_neighbor, a2_neighbor, a2_m_a1_neighbor, m_a1_neighbor, m_a2_neighbor, a1_m_a2_neighbor]
        return result

if __name__ == "__main__":

    print(find_NNN(0))