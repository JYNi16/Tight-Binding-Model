import numpy as np
import matplotlib.pyplot as plt
from Haldane_collection.Lattice import Lattice_2D as Lattice
a = 2.46


def angle_between(v1, v2):
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


lat1 = Lattice(lattice_vector=np.array([[np.sqrt(3) / 2, -1 / 2, 0],
                                        [np.sqrt(3) / 2, 1 / 2, 0],
                                        [0, 0, 1]], dtype=np.float64).T * a,
               theta=0,
               high_symmetry_points=np.array([[0, 0],  # Gamma
                                              [1 / 3, 2 / 3],  # K
                                              [1 / 2, 1 / 2],  # M
                                              [2 / 3, 1 / 3],  # K_prime
                                              [0, 0]], dtype=np.float64).T,
               high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'])

lat2 = Lattice(lattice_vector=np.array([[1, 0, 0],
                                        [1 / 2, np.sqrt(3) / 2, 0],
                                        [0, 0, 1]], dtype=np.float64).T * a * np.sqrt(3),
               theta=45,
               high_symmetry_points=np.array([[0, 0],  # Gamma
                                              [1 / 3, 2 / 3],  # K
                                              [1 / 2, 1 / 2],  # M
                                              [2 / 3, 1 / 3],  # K_prime
                                              [0, 0]], dtype=np.float64).T,
               high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'])

lat3 = Lattice(lattice_vector=np.array([[np.sqrt(3) / 2, - 1 / 2, 0],
                                        [np.sqrt(3) / 2, 1 / 2, 0],
                                        [0, 0, 1]], dtype=np.float64).T * a,
               theta=0,
               high_symmetry_points=np.array([[0, 0],  # Gamma
                                              [1 / 3, 2 / 3],  # K
                                              [1 / 2, 1 / 2],  # M
                                              [2 / 3, 1 / 3],  # K_prime
                                              [0, 0]], dtype=np.float64).T,
               high_symmetry_points_labels=['$\Gamma$', 'K', 'M', '$K\'$', '$\Gamma$'])

lat = lat3

plt.figure(
    figsize=(10.0, 5.0),
    dpi=300
)

axes = [plt.subplot(121), plt.subplot(122)]
ax = axes[0]
ax.arrow(0, 0, lat.a_1[0], lat.a_1[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
ax.arrow(0, 0, lat.a_2[0], lat.a_2[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
ax.text(lat.a_1[0], lat.a_1[1], r'$\vec{a}_1$', fontsize=20)
ax.text(lat.a_2[0], lat.a_2[1], r'$\vec{a}_2$', fontsize=20)
# ax.set_xlim(-real_range, real_range)
# ax.set_ylim(-real_range, real_range)
ax.set_aspect('equal')
ax.grid(True, which='both')
ax.set_title(f"Lattice: {lat.lattice_vector[:2, :2]}")

# plt.arrow(0, 0, a_1[0], a_1[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
# plt.arrow(0, 0, a_2[0], a_2[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
# plt.text(a_1[0], a_1[1], r'$\vec{a}_1$', fontsize=20)
# plt.text(a_2[0], a_2[1], r'$\vec{a}_2$', fontsize=20)

ax = axes[1]
ax.arrow(0, 0, lat.b_1[0], lat.b_1[1], head_width=0.02, head_length=0.02, fc='b', ec='b')
ax.arrow(0, 0, lat.b_2[0], lat.b_2[1], head_width=0.02, head_length=0.02, fc='b', ec='b')
ax.text(lat.b_1[0], lat.b_1[1], r'$\vec{b}_1$', fontsize=20)
ax.text(lat.b_2[0], lat.b_2[1], r'$\vec{b}_2$', fontsize=20)

length = np.linalg.norm(lat.b_1) / np.sqrt(3)

# plot hexagon
a = (1 + 0j) * length * np.exp((angle_between(lat.b_1, lat.b_2) - 1.0471975511965979 + lat.theta) / 2 * 1j)
cc = np.exp(np.linspace(0, 2 * np.pi, 7) * 1j) * a
xx = cc.real
yy = cc.imag
ax.plot(xx, yy, 'k-')
ax.plot(xx + lat.b_1[0], yy + lat.b_1[1], 'k-')
ax.plot(xx + lat.b_2[0], yy + lat.b_2[1], 'k-')

for ind, high_symmetry_point in enumerate(lat.high_symmetry_points_cart.T):
    # print(high_symmetry_point, labels[ind], ind)
    ax.plot(lat.high_symmetry_points_cart[0, ind], lat.high_symmetry_points_cart[1, ind], 'ro', markersize=10)
    ax.text(lat.high_symmetry_points_cart[0, ind], lat.high_symmetry_points_cart[1, ind],
            lat.high_symmetry_points_labels[ind], fontsize=20)

for ind, high_symmetry_point in enumerate(lat2.high_symmetry_points_cart.T):
    # print(high_symmetry_point, labels[ind], ind)
    ax.plot(high_symmetry_point[0], high_symmetry_point[1], 'bo', markersize=10)
    ax.text(high_symmetry_point[0], high_symmetry_point[1],
            lat.high_symmetry_points_labels[ind], fontsize=20)




ax.set_aspect('equal')
ax.grid(True, which='both')
ax.set_title(f"Reciprocal lattice: {lat.rlat[:2, :2]}")
plt.show()
