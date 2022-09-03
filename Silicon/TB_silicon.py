import numpy as np

def phase(k, neighbors):
    '''
    Determines phase factors of overlap parameters using the assumption that the
    orbitals of each crystal overlap only with those of its nearest neighbor.

    args:
        k: A numpy array of shape (3,) that represents the k-point at which to
           calculate phase factors.
        neighbors: A numpy array of shape (4, 3) that represents the four nearest
                   neighbors in the lattice of an atom centered at (0, 0, 0).

    returns:
        A numpy array of shape (4,) containing the (complex) phase factors.
    '''

    a, b, c, d = [np.exp(1j * k @ neighbor) for neighbor in neighbors]
    factors = np.array([
        a + b + c + d,
        a + b - c - d,
        a - b + c - d,
        a - b - c + d
    ])
    return (1 / 4) * factors


def band_energies(g, es, ep, vss, vsp, vxx, vxy):
    '''
    Calculates the band energies (eigenvalues) of a material using the
    tight-binding approximation for single nearest-neighbor interactions.

    args:
        g: A numpy array of shape (4,) representing the phase factors with respect
           to a wavevector k and the crystal's nearest neighbors.
        es, ep, vss, vsp, vxx, vxy: Empirical parameters for orbital overlap
                                    interactions between nearest neighbors.

    returns:
        A numpy array of shape (8,) containing the eigenvalues of the
        corresponding Hamiltonian.
    '''

    gc = np.conjugate(g)

    hamiltonian = np.array([
        [es, vss * g[0], 0, 0, 0, vsp * g[1], vsp * g[2], vsp * g[3]],
        [vss * gc[0], es, -vsp * gc[1], -vsp * gc[2], -vsp * gc[3], 0, 0, 0],
        [0, -vsp * g[1], ep, 0, 0, vxx * g[0], vxy * g[3], vxy * g[1]],
        [0, -vsp * g[2], 0, ep, 0, vxy * g[3], vxx * g[0], vxy * g[1]],
        [0, -vsp * g[3], 0, 0, ep, vxy * g[1], vxy * g[2], vxx * g[0]],
        [vsp * gc[1], 0, vxx * gc[0], vxy * gc[3], vxy * gc[1], ep, 0, 0],
        [vsp * gc[2], 0, vxy * gc[3], vxx * gc[0], vxy * gc[2], 0, ep, 0],
        [vsp * gc[3], 0, vxy * gc[1], vxy * gc[1], vxx * gc[0], 0, 0, ep]
    ])

    eigvals = np.linalg.eigvalsh(hamiltonian)
    eigvals.sort()
    return eigvals


def band_structure(params, neighbors, path):
    bands = []

    for k in np.vstack(path):
        g = phase(k, neighbors)
        eigvals = band_energies(g, *params)
        bands.append(eigvals)

    return np.stack(bands, axis=-1)


def linpath(a, b, n=50, endpoint=True):
    '''
    Creates an array of n equally spaced points along the path a -> b, not inclusive.

    args:
        a: An iterable of numbers that represents the starting position.
        b: An iterable of numbers that represents the ending position.
        n: The integer number of sample points to calculate. Defaults to 50.

    returns:
        A numpy array of shape (n, k) where k is the shortest length of either
        iterable -- a or b.
    '''
    # list of n linear spacings between the start and end of each corresponding point
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(a, b)]

    # stacks along their last axis, transforming a list of spacings into an array of points of len n
    return np.stack(spacings, axis=-1)

