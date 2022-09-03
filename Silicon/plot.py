import numpy as np
from TB_silicon import *


# Es, Ep, Vss, Vsp, Vxx, Vxy
# Ep - Es = 7.20
params = (-4.03, 3.17, -8.13, 5.88, 3.17, 7.51)

# k-points per path
n = 1000

# lattice constant
a = 1

# nearest neighbors to atom at (0, 0, 0)
neighbors = a / 4 *  np.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1]
])

# symmetry points in the Brillouin zone
G = 2 * np.pi / a * np.array([0, 0, 0])
L = 2 * np.pi / a * np.array([1/2, 1/2, 1/2])
K = 2 * np.pi / a * np.array([3/4, 3/4, 0])
X = 2 * np.pi / a * np.array([0, 0, 1])
W = 2 * np.pi / a * np.array([1, 1/2, 0])
U = 2 * np.pi / a * np.array([1/4, 1/4, 1])

# k-paths
lambd = linpath(L, G, n, endpoint=False)
delta = linpath(G, X, n, endpoint=False)
x_uk = linpath(X, U, n // 4, endpoint=False)
sigma = linpath(K, G, n, endpoint=True)

bands = band_structure(params, neighbors, path=[lambd, delta, x_uk, sigma])
# k-paths
lambd = linpath(L, G, n, endpoint=False)
delta = linpath(G, X, n, endpoint=False)
x_uk = linpath(X, U, n // 4, endpoint=False)
sigma = linpath(K, G, n, endpoint=True)

from matplotlib import pyplot as plt

plt.figure(figsize=(8, 6))

ax = plt.subplot(111)

# remove plot borders
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)

# limit plot area to data
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

# custom tick names for k-points
xticks = n * np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$K$', '$\Sigma$', '$\Gamma$'), fontsize=18)
plt.yticks(fontsize=18)

# horizontal guide lines every 2.5 eV
# for y in np.arange(-25, 25, 2.5):
#     plt.axhline(y, ls='--', lw=0.3, color='black', alpha=0.3)

# hide ticks, unnecessary with gridlines
# plt.tick_params(axis='both', which='both',
#                 top='off', bottom='off', left='off', right='off',
#                 labelbottom='on', labelleft='on', pad=5)

plt.xlabel('k-Path', fontsize=20)
plt.ylabel('Energy (eV)', fontsize=20)
# plt.text(1350, -18, 'Fig. 1. Band structure of Si.', fontsize=12)

# tableau 10 in fractional (r, g, b)
colors = 1 / 255 * np.array([
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207]
])

for band, color in zip(bands, colors):
    plt.plot(band, lw=2.0, color=color)

#plot fermi level
plt.hlines(y = 0, xmin = 0, xmax = len(bands))

plt.savefig("Figure/band_si.png", dpi=300)
plt.show()