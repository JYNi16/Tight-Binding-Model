# from Lattice import graphene_60_lat as lat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg', force=True)
t1 = 1
N_k = 100

a = 1.42
nn1 = np.array([-1, 0])
nn2 = np.array([1 / 2, -np.sqrt(3) / 2])
nn3 = np.array([1 / 2, np.sqrt(3) / 2])

a_1 = np.array([np.sqrt(3) / 2, -1 / 2])
a_2 = np.array([np.sqrt(3) / 2, 1 / 2])

# a = 2.46

# d = 2.46 / np.sqrt(3)

all_kx = np.linspace(-np.pi, np.pi, N_k)
all_ky = np.linspace(-np.pi, np.pi, N_k)
kx, ky = np.meshgrid(all_kx, all_ky)

all_eigv_atom = np.zeros((N_k, N_k, 2), dtype=np.double)

for i in range(N_k):
    for j in range(N_k):
        k = np.array([all_kx[i], all_ky[j]])
        H = np.zeros((2, 2), dtype=complex)
        H[0, 1] = t1 * (np.exp(1j * k @ nn1) +
                        np.exp(1j * k @ nn2) +
                        np.exp(1j * k @ nn3))

        H[1, 0] = np.conj(H[0, 1])
        eigv = np.linalg.eigvalsh(H)
        all_eigv_atom[i, j, :] = eigv

all_eigv_transform = np.zeros((N_k, N_k, 2), dtype=np.double)

for i in range(N_k):
    for j in range(N_k):
        k = np.array([all_kx[i], all_ky[j]])
        H = np.zeros((2, 2), dtype=complex)
        H[0, 1] = -t1 * (1 +
                         np.exp(1j * k @ a_1) +
                         np.exp(1j * k @ a_2))

        H[1, 0] = np.conj(H[0, 1])
        eigv = np.linalg.eigvalsh(H)
        all_eigv_transform[i, j, :] = eigv

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

surf = ax.plot_surface(kx, ky,
                       # all_eigv_atom[:, :, 0],
                       all_eigv_transform[:, :, 0],
                       cmap='viridis',
                       linewidth=0,
                       antialiased=False)

surf2 = ax.plot_surface(kx, ky,
                        # all_eigv_atom[:, :, 1],
                        all_eigv_transform[:, :, 1],
                        cmap='viridis',
                        linewidth=0,
                        antialiased=False)

print("Switched to:", matplotlib.get_backend())
plt.show()
