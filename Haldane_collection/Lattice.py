import numpy as np
import band_ini.k_sym_gen as ksg


class Lattice_2D:
    def cal_reciprocal_lattice(self):
        C = 2 * np.pi
        rlat = np.linalg.inv(self.lattice_vector).T * C
        self.rlat = rlat

    def __init__(self, lattice_vector, theta, high_symmetry_points, high_symmetry_points_labels):
        self.lattice_vector = lattice_vector
        self.theta = theta
        self.high_symmetry_points = high_symmetry_points
        self.rlat = None
        self.high_symmetry_points_labels = high_symmetry_points_labels
        self.cal_reciprocal_lattice()
        self.high_symmetry_points_cart = self.rlat[:2, :2] @ self.high_symmetry_points
        self.a_1 = self.lattice_vector[:2, 0]
        self.a_2 = self.lattice_vector[:2, 1]
        self.b_1 = self.rlat[:2, 0]
        self.b_2 = self.rlat[:2, 1]
        self.all_high_sym_points = list(self.high_symmetry_points_cart.T)
        self.k_point_path, self.k_path, self.Node = ksg.k_path_sym_gen(self.all_high_sym_points)
        self.k_point_path = np.array(self.k_point_path).reshape((len(self.all_high_sym_points) - 1) * 100, 2)


a = 2.46
graphene_60_lat = Lattice_2D(lattice_vector=np.array([[np.sqrt(3) / 2, -1 / 2, 0],
                                                      [np.sqrt(3) / 2, 1 / 2, 0],
                                                      [0, 0, 1]], dtype=np.float64).T * a,
                             theta=0,
                             high_symmetry_points=np.array([[0, 0],  # Gamma
                                                            [1 / 3, 2 / 3],  # K
                                                            [1 / 2, 1 / 2],  # M
                                                            [2 / 3, 1 / 3],  # K_prime
                                                            [0, 0]], dtype=np.float64).T,
                             high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'])

if __name__ == "__main__":
    a = 2.46
    lat2 = Lattice_2D(lattice_vector=np.array([[1, 0, 0],
                                               [1 / 2, np.sqrt(3) / 2, 0],
                                               [0, 0, 1]], dtype=np.float64).T * a * np.sqrt(3),
                      theta=45,
                      high_symmetry_points=np.array([[0, 0],  # Gamma
                                                     [1 / 3, 2 / 3],  # K
                                                     [1 / 2, 1 / 2],  # M
                                                     [2 / 3, 1 / 3],  # K_prime
                                                     [0, 0]], dtype=np.float64).T,
                      high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'])
