import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#setting parameters
N = 400  # 
t1 = 1.0  # NN hopping
t2 = 1.2  # NNN hopping

# Hamiltonian =
H = np.zeros((2*N, 2*N))  
for i in range(N):
    # Intracell hopping (within a unit cell)
    H[2*i, 2*i+1] = t1
    H[2*i+1, 2*i] = t1
    # Intercell hopping (between adjacent unit cells)
    if i < N-1:
        H[2*i+1, 2*i+2] = t2
        H[2*i+2, 2*i+1] = t2

#solving eigenvalues and eigenvectors
eigenvalues, eigenvectors = la.eigh(H)

#Band 
plt.figure(figsize=(8, 6))
plt.plot(eigenvalues, 'bo', label='Eigenvalues')
plt.axhline(0, color='r', linestyle='--', label='Zero energy')
plt.title('Eigenvalues of SSH Model (Open Boundary Condition)')
plt.xlabel('Index')
plt.ylabel('Energy')
#plt.xlim(300, 500)
#plt.ylim(-0.5, 0.5)
plt.legend()
plt.show()

# seeking the corner states
tolerance = 1e-5  # tolerance
zero_energy_indices = np.where(np.abs(eigenvalues) < tolerance)[0]
corner_states = eigenvectors[:, zero_energy_indices]

# Print the eigenvalues closed to zero-energy
print("接近零的本征值:", eigenvalues[zero_energy_indices])
print("角态对应的本征向量:")
print(corner_states)

# print the probility
plt.figure(figsize=(8, 6))
for i, state in enumerate(corner_states.T):
    prob_density = np.abs(state)**2
    plt.plot(prob_density, label=f'Corner State {i+1}')
plt.title('Probability Distribution of Corner States')
plt.xlabel('Site index')
plt.ylabel('Probability')
plt.legend()
plt.show()



