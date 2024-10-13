import matplotlib.pyplot as plt
import numpy as np


def plot_band(all_eigv, lat):
    plt.figure()
    for i in range(np.shape(all_eigv)[0]):
        plt.plot(lat.k_path, all_eigv[i, :], 'k')
    plt.xticks(lat.Node, lat.high_symmetry_points_labels)
    plt.show()
